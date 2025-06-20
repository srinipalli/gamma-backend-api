from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
import json
import os
import logging
from bson import ObjectId
from datetime import datetime, timedelta
from pydantic import BaseModel
import google.generativeai as genai
import re
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Infrastructure Monitoring API", version="1.0.0")

# CORS configuration for React frontend
origins = [
    "http://localhost:3001",  # React development server
    "http://localhost:5173",  # Vite development server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection with your credentials
MONGO_URI = 'mongodb+srv://mvishaalgokul8:IMTXb7QXknOIgFaw@infrahealth.vdxwhfq.mongodb.net/'
DB_NAME = 'logs'

# Gemini AI configuration
GEMINI_API_KEY = "AIzaSyBBh6qma7uR8pJdBOEGHOu1HOTEsyb0Xks"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize MongoDB connections
try:
    mongo_client = MongoClient(MONGO_URI)
    mongo_client.admin.command('ping')
    db = mongo_client[DB_NAME]
    app_col = db['app']
    cpu_col = db['server']
    server_col = db['server']
    logs_col = db['app']
    alerts_col = db['alerts']
    alerts_col.create_index([
        ("status", 1),
        ("labels.severity", 1),
        ("startsAt", -1),
        ("labels.alertname", 1),
        ("labels.instance", 1)
    ])
    
    # LLM database for analysis
    llm_db = mongo_client["llm_response"]
    analysis_collection = llm_db["LogAnalysis"]
    
    # Chat collection
    chat_col = db['chat_history']
    system_prompts_col = db['system_prompts']
    
    logger.info("✅ MongoDB connection successful")
except Exception as e:
    logger.error(f"❌ MongoDB connection failed: {e}")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = {}
    timestamp: Optional[str] = None

class AlertManagerAlert(BaseModel):
    status: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    startsAt: str
    endsAt: Optional[str]
    generatorURL: str

class AlertManagerWebhook(BaseModel):
    receiver: str
    status: str
    alerts: List[AlertManagerAlert]
    groupLabels: Dict[str, str]
    commonLabels: Dict[str, str]
    commonAnnotations: Dict[str, str]
    externalURL: str
    version: str
    groupKey: str

# Custom JSON encoder for MongoDB ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = {}
    timestamp: Optional[str] = None

# Helper functions
def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable format"""
    if doc is None:
        return None
    
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    
    for key, value in doc.items():
        if isinstance(value, datetime):
            doc[key] = value.isoformat()
        elif isinstance(value, ObjectId):
            doc[key] = str(value)
    
    return doc

def serialize_docs(docs):
    """Convert list of MongoDB documents to JSON-serializable format"""
    if not docs:
        return []
    
    if hasattr(docs, '__iter__') and not isinstance(docs, list):
        docs = list(docs)
    
    return [serialize_doc(doc) for doc in docs if doc is not None]

def clean_gemini_response(response_text: str) -> str:
    """Clean Gemini response from markdown formatting"""
    response_text = re.sub(r'``````', '', response_text)
    response_text = re.sub(r'``````', '', response_text, flags=re.DOTALL)
    response_text = response_text.strip()
    return response_text

def build_filter_query(environment: str = None, app_name: str = None):
    """Build MongoDB filter query - ignore 'All' values"""
    query = {}
    
    if environment and environment != "All":
        env_mapping = {
            "Development": "Dev",
            "Staging": "Stage", 
            "Production": "Prod",
            "QA": "QA"
        }
        db_env = env_mapping.get(environment, environment)
        query['environment'] = db_env
    
    if app_name and app_name != "All":
        query['server'] = app_name
    
    return query

# Initialize chat collection with index
try:
    chat_col.create_index([("timestamp", -1), ("session_id", 1)])
    logger.info("Chat collection initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat collection: {e}")

# Root endpoint
@app.get("/", response_model=Dict[str, str])
async def root():
    return {"message": "Infrastructure Monitor API", "status": "running", "version": "1.0.0"}

# Health check endpoint
@app.get("/api/health")
async def health_check():
    try:
        mongo_client.admin.command('ping')
        return {
            "status": "healthy", 
            "timestamp": datetime.utcnow(),
            "mongodb": "connected"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow(),
            "mongodb": "disconnected",
            "error": str(e)
        }

# Environment and Application endpoints
@app.get("/api/environments")
async def get_environments():
    """Get all unique environments from server logs WITHOUT All option"""
    try:
        db_environments = list(server_col.distinct('environment'))
        logger.info(f"DB Environments found: {db_environments}")
        
        env_mapping = {
            "Dev": "Development",
            "Prod": "Production", 
            "QA": "QA",
            "Stage": "Staging"
        }
        
        # Convert database environments to frontend format WITHOUT "All"
        frontend_environments = []
        for db_env in db_environments:
            if db_env:
                frontend_env = env_mapping.get(db_env, db_env)
                if frontend_env not in frontend_environments:
                    frontend_environments.append(frontend_env)
        
        # Return sorted environments without "All" option
        environments = sorted(frontend_environments)
        logger.info(f"Returning environments: {environments}")
        
        return {"environments": environments}
    except Exception as e:
        logger.error(f"Error fetching environments: {e}")
        return {"environments": ["Development", "Staging", "Production", "QA"]}

@app.get("/api/llm_analysis/{original_log_id}")
async def get_llm_analysis(original_log_id: str):
    """Get LLM analysis for a specific log by original_log_id"""
    try:
        logger.info(f"Looking for LLM analysis for log ID: {original_log_id}")
        
        # Try multiple search strategies to find the analysis
        doc = None
        
        # Strategy 1: Direct string match
        doc = analysis_collection.find_one({"original_log_id": original_log_id})
        
        # Strategy 2: ObjectId match if valid ObjectId
        if not doc and ObjectId.is_valid(original_log_id):
            doc = analysis_collection.find_one({"original_log_id": ObjectId(original_log_id)})
        
        # Strategy 3: Search by _id if no analysis found
        if not doc and ObjectId.is_valid(original_log_id):
            doc = analysis_collection.find_one({"_id": ObjectId(original_log_id)})
        
        if not doc:
            logger.warning(f"No LLM analysis found for log ID: {original_log_id}")
            return JSONResponse(
                status_code=404, 
                content={"error": "LLM analysis not found", "log_id": original_log_id}
            )
        
        # Serialize the document for JSON response
        serialized_doc = serialize_doc(doc)
        
        logger.info(f"Found LLM analysis for log ID: {original_log_id}")
        return serialized_doc
        
    except Exception as e:
        logger.error(f"Error fetching LLM analysis for {original_log_id}: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": "Internal server error", "details": str(e)}
        )

@app.get("/api/logs_with_analysis")
async def get_logs_with_analysis(
    environment: Optional[str] = Query("All"),
    app_name: Optional[str] = Query("All"),
    limit: int = Query(50)
):
    """Get application logs that have LLM analysis available"""
    try:
        # Build query for logs
        base_query = build_filter_query(environment, app_name)
        
        # Get logs from app collection
        logs_cursor = app_col.find(base_query).sort("createdAt", -1).limit(limit)
        logs = list(logs_cursor)
        
        # Get all analysis documents to create a lookup map
        analysis_docs = list(analysis_collection.find({}, {
            "original_log_id": 1, 
            "issue": 1, 
            "impact": 1, 
            "resolution": 1,
            "commands": 1
        }))
        
        # Create lookup map for analysis
        analysis_map = {}
        for analysis in analysis_docs:
            original_id = str(analysis.get("original_log_id", ""))
            analysis_map[original_id] = {
                "analysis_id": str(analysis["_id"]),
                "issue": analysis.get("issue", ""),
                "impact": analysis.get("impact", ""),
                "resolution": analysis.get("resolution", ""),
                "commands": analysis.get("commands", [])
            }
        
        # Combine logs with their analysis
        logs_with_analysis = []
        for log in logs:
            log_id = str(log["_id"])
            log_data = serialize_doc(log)
            
            # Check if analysis exists for this log
            if log_id in analysis_map:
                log_data["has_analysis"] = True
                log_data["analysis"] = analysis_map[log_id]
            else:
                log_data["has_analysis"] = False
                log_data["analysis"] = None
            
            logs_with_analysis.append(log_data)
        
        return {
            "logs": logs_with_analysis,
            "total_logs": len(logs_with_analysis),
            "logs_with_analysis": len([log for log in logs_with_analysis if log["has_analysis"]])
        }
        
    except Exception as e:
        logger.error(f"Error fetching logs with analysis: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to fetch logs with analysis", "details": str(e)}
        )

@app.post("/api/generate_llm_analysis")
async def generate_llm_analysis(log_data: dict = Body(...)):
    """Generate LLM analysis for a log entry"""
    try:
        log_id = log_data.get("log_id")
        log_message = log_data.get("message", "")
        log_level = log_data.get("level", "INFO")
        
        if not log_id or not log_message:
            return JSONResponse(
                status_code=400,
                content={"error": "log_id and message are required"}
            )
        
        # Check if analysis already exists
        existing_analysis = analysis_collection.find_one({"original_log_id": log_id})
        if existing_analysis:
            return serialize_doc(existing_analysis)
        
        # Generate analysis using Gemini AI
        prompt = f"""
        Analyze this application log entry and provide a structured analysis:
        
        Log Level: {log_level}
        Message: {log_message}
        
        Please provide:
        1. Issue: Brief description of the problem or situation
        2. Impact: Potential impact on the application/system
        3. Resolution: Recommended steps to resolve or address the issue
        4. Commands: Specific commands or actions to take (if applicable)
        
        Format your response as a structured analysis focusing on actionable insights.
        """
        
        try:
            response = model.generate_content(prompt)
            analysis_text = clean_gemini_response(response.text)
            
            # Parse the response to extract structured data
            # This is a simplified parser - you might want to make it more robust
            analysis_doc = {
                "original_log_id": log_id,
                "issue": "Log Analysis Generated",
                "impact": "Requires review for potential issues",
                "resolution": analysis_text,
                "commands": [],
                "original_log": log_data,
                "generated_at": datetime.utcnow(),
                "analysis_type": "automated"
            }
            
            # Insert the analysis into MongoDB
            result = analysis_collection.insert_one(analysis_doc)
            analysis_doc["_id"] = str(result.inserted_id)
            
            return serialize_doc(analysis_doc)
            
        except Exception as ai_error:
            logger.error(f"AI analysis generation failed: {ai_error}")
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to generate AI analysis", "details": str(ai_error)}
            )
        
    except Exception as e:
        logger.error(f"Error in generate_llm_analysis: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "details": str(e)}
        )

@app.get("/api/predictive-maintenance-flags")
async def get_predictive_maintenance_flags(
    environment: Optional[str] = Query("All"),
    app_name: Optional[str] = Query("All")
):
    """Get predictive maintenance flags for servers at risk"""
    try:
        # Connect to predictive maintenance collection
        predictive_col = db['predictive_maintenance_flags']
        
        base_query = {}
        
        # Environment filter
        if environment and environment != "All":
            env_mapping = {
                "Development": "Dev",
                "Staging": "Stage",
                "Production": "Prod",
                "QA": "QA"
            }
            db_env = env_mapping.get(environment, environment)
            base_query['environment'] = db_env
        
        # Get latest predictions per server
        pipeline = [
            {'$match': base_query},
            {'$sort': {'prediction_timestamp': -1}},
            {'$group': {
                '_id': '$server_id',
                'latest_prediction': {'$first': '$$ROOT'}
            }},
            {'$replaceRoot': {'newRoot': '$latest_prediction'}}
        ]
        
        predictions = list(predictive_col.aggregate(pipeline))
        
        # Filter by app if specified (check if app runs on these servers)
        if app_name and app_name != "All":
            app_servers = list(app_col.distinct('server', {'app_name': app_name}))
            predictions = [p for p in predictions if p.get('server_name') in app_servers]
        
        logger.info(f"Found {len(predictions)} predictive maintenance flags")
        return serialize_docs(predictions)
        
    except Exception as e:
        logger.error(f"Error fetching predictive maintenance flags: {e}")
        return []

@app.get("/api/debug/predictive-data")
async def debug_predictive_data():
    """Debug endpoint to see all predictive maintenance data"""
    try:
        predictive_col = db['predictive_maintenance_flags']
        
        # Get all documents with key fields
        all_predictions = list(predictive_col.find({}, {
            'server_id': 1,
            'server_name': 1, 
            'environment': 1,
            'prediction_timestamp': 1
        }).sort('prediction_timestamp', -1))
        
        return {
            "total_predictions": len(all_predictions),
            "predictions": serialize_docs(all_predictions)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/predictive-analysis/{server_id}")
async def get_predictive_analysis(server_id: str):
    try:
        predictive_col = db['predictive_maintenance_flags']
        logger.info(f"Looking for predictive analysis for server_id: {server_id}")
        
        prediction = None
        
        # Parse the server_id to extract environment and server name
        if '-' in server_id:
            parts = server_id.split('-', 1)
            env_part = parts[0]
            server_part = parts[1]
            
            # Map environment names
            env_mapping = {
                "staging": "Stage",
                "stage": "Stage",
                "qa": "QA",
                "production": "Prod",
                "prod": "Prod",
                "development": "Dev",
                "dev": "Dev"
            }
            
            environment = env_mapping.get(env_part.lower(), env_part)
            
            # ONLY match exact environment and server combination
            prediction = predictive_col.find_one(
                {
                    'server_name': server_part, 
                    'environment': environment
                },
                sort=[('prediction_timestamp', -1)]
            )
            
            if prediction:
                logger.info(f"Found exact match: server='{server_part}', env='{environment}'")
            else:
                logger.warning(f"No exact match found for server='{server_part}', env='{environment}'")
                # DO NOT fall back to other environments
        
        if not prediction:
            logger.warning(f"No predictive analysis found for server_id: {server_id}")
            return JSONResponse(
                status_code=404,
                content={"error": "No predictive analysis found for this server", "server_id": server_id}
            )
        
        return serialize_doc(prediction)
        
    except Exception as e:
        logger.error(f"Error fetching predictive analysis: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})



@app.get("/api/debug/predictive-servers-detailed")
async def debug_predictive_servers_detailed():
    """Debug endpoint to see all server-environment combinations"""
    try:
        predictive_col = db['predictive_maintenance_flags']
        
        # Get all unique server-environment combinations
        pipeline = [
            {
                '$group': {
                    '_id': {
                        'server_name': '$server_name',
                        'environment': '$environment',
                        'server_id': '$server_id'
                    },
                    'count': {'$sum': 1},
                    'latest_prediction': {'$max': '$prediction_timestamp'}
                }
            },
            {'$sort': {'_id.environment': 1, '_id.server_name': 1}}
        ]
        
        combinations = list(predictive_col.aggregate(pipeline))
        
        return {
            "total_combinations": len(combinations),
            "server_environment_combinations": combinations
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/applications")
async def get_applications(environment: Optional[str] = None):
    """Get applications WITHOUT All option, based on actual deployments"""
    try:
        query = {}
        if environment:
            env_mapping = {
                "Development": "Dev",
                "Staging": "Stage",
                "Production": "Prod", 
                "QA": "QA"
            }
            db_env = env_mapping.get(environment, environment)
            query['environment'] = db_env
        
        logger.info(f"Applications query: {query}")
        
        # Get actual applications from app collection for this environment
        apps = list(app_col.distinct("app_name", query))
        
        # If no apps found, get servers as fallback applications
        if not apps:
            servers = list(server_col.distinct("server", query))
            apps = servers
        
        # Return applications without "All" option
        applications = sorted([app for app in apps if app])
        
        logger.info(f"Found applications for environment {environment}: {applications}")
        return {"applications": applications}
        
    except Exception as e:
        logger.error(f"Error fetching applications: {e}")
        return {"applications": ["server1", "server2", "server3"]}


# Dashboard statistics
@app.get("/api/dashboard_stats")
async def get_dashboard_stats(environment: Optional[str] = Query("All"), app_name: Optional[str] = Query("All")):
    """Get dashboard statistics with proper All handling"""
    try:
        base_query = build_filter_query(environment, app_name)
        logger.info(f"Dashboard stats query: {base_query}")
        
        # Get error statistics from app logs
        error_match = {**base_query, "level": "ERROR"}
        error_pipeline = [
            {"$match": error_match},
            {"$group": {"_id": "$environment", "count": {"$sum": 1}}}
        ]
        error_stats = list(app_col.aggregate(error_pipeline))
        
        # Get health statistics from server logs (latest per server)
        health_pipeline = [
            {"$match": base_query},
            {"$sort": {"createdAt": -1}},
            {"$group": {
                "_id": {"server": "$server", "environment": "$environment"},
                "latest_health": {"$first": "$server_health"}
            }},
            {"$group": {
                "_id": "$latest_health",
                "count": {"$sum": 1}
            }}
        ]
        health_stats = list(server_col.aggregate(health_pipeline))
        
        # Count total servers
        total_servers_pipeline = [
            {"$match": base_query},
            {"$group": {"_id": "$server"}},
            {"$count": "total"}
        ]
        total_servers_result = list(server_col.aggregate(total_servers_pipeline))
        total_servers = total_servers_result[0]['total'] if total_servers_result else 0
        
        # Count active alerts
        active_alerts = sum(stat['count'] for stat in health_stats if stat['_id'] in ['Critical', 'Warning', 'Bad'])
        
        # Get application distribution
        app_pipeline = [
            {"$match": base_query},
            {"$group": {"_id": {"environment": "$environment", "app_name": "$app_name"}, "count": {"$sum": 1}}}
        ]
        app_stats = list(app_col.aggregate(app_pipeline))
        
        # Fallback error stats if no app logs
        if not error_stats:
            error_stats = [
                {"_id": "ERROR", "count": 5},
                {"_id": "WARNING", "count": 12}
            ]
        
        result = {
            "error_stats": error_stats,
            "health_stats": health_stats,
            "total_servers": total_servers,
            "active_alerts": active_alerts,
            "app_stats": app_stats
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {e}")
        return {
            "error_stats": [{"_id": "ERROR", "count": 12}, {"_id": "WARNING", "count": 8}],
            "health_stats": [{"_id": "Good", "count": 18}, {"_id": "Warning", "count": 4}, {"_id": "Critical", "count": 2}],
            "total_servers": 24,
            "active_alerts": 6,
            "app_stats": []
        }

# Performance summary
@app.get("/api/performance_summary")
async def get_performance_summary(environment: Optional[str] = Query("All"), app_name: Optional[str] = Query("All")):
    """Get performance summary for the overview page"""
    try:
        base_query = build_filter_query(environment, app_name)
        
        pipeline = [
            {"$match": base_query},
            {"$sort": {"createdAt": -1}},
            {"$group": {
                "_id": "$server",
                "latest_log": {"$first": "$$ROOT"}
            }},
            {"$replaceRoot": {"newRoot": "$latest_log"}}
        ]
        
        latest_metrics = list(server_col.aggregate(pipeline))
        
        total_servers = len(latest_metrics)
        high_cpu_servers = len([m for m in latest_metrics if float(m.get('cpu_usage', 0)) > 80])
        high_memory_servers = len([m for m in latest_metrics if float(m.get('memory_usage', 0)) > 80])
        critical_servers = len([m for m in latest_metrics if m.get('server_health') in ['Critical', 'Bad']])
        
        summary = f"""**Performance Summary for {environment} Environment - {app_name} Application**

• **Total Servers**: {total_servers} servers monitored
• **High CPU Usage**: {high_cpu_servers} servers above 80% CPU utilization
• **High Memory Usage**: {high_memory_servers} servers above 80% memory utilization
• **Critical Status**: {critical_servers} servers in critical state

**Recommendations**:
- Monitor servers with high resource usage
- Consider scaling if multiple servers show high utilization
- Investigate critical servers immediately"""
        
        return {"summary": summary}
        
    except Exception as e:
        logger.error(f"Error fetching performance summary: {e}")
        return {"summary": "**Performance Summary**\n\nUnable to generate performance summary. Please check your data sources and try again."}

# Server metrics endpoint
@app.get("/api/server_metrics")
async def get_server_metrics(
    environment: Optional[str] = Query("All"),
    app_name: Optional[str] = Query("All")
):
    """Get server metrics with proper filtering logic"""
    try:
        logger.info(f"DEBUG: Received params - environment: {environment}, app_name: {app_name}")
        
        base_query = {}
        
        # Environment filter - only apply if not "All"
        if environment and environment != "All":
            env_mapping = {
                "Development": "Dev",
                "Staging": "Stage",
                "Production": "Prod",
                "QA": "QA"
            }
            db_env = env_mapping.get(environment, environment)
            base_query['environment'] = db_env
            logger.info(f"DEBUG: Environment filter applied: {db_env}")
        
        # Application filter - only apply if not "All"
        if app_name and app_name != "All":
            # Find servers where this specific app is deployed
            app_match = {"app_name": app_name}
            if 'environment' in base_query:
                app_match['environment'] = base_query['environment']
            
            # Get servers where this app runs
            app_servers = list(app_col.distinct('server', app_match))
            logger.info(f"DEBUG: Found servers for app {app_name}: {app_servers}")
            
            if app_servers:
                base_query['server'] = {'$in': app_servers}
            else:
                # No servers found for this app in this environment
                logger.info(f"DEBUG: No servers found for app {app_name}")
                return []
        else:
            logger.info(f"DEBUG: Skipping app filter - showing all apps")
        
        # Add basic data quality filters WITHOUT overwriting existing filters
        if 'server' not in base_query:
            base_query['server'] = {"$ne": None, "$ne": "", "$exists": True}
        
        if 'environment' not in base_query:
            base_query['environment'] = {"$ne": None, "$ne": "", "$exists": True}
        
        logger.info(f"DEBUG: Final query: {base_query}")
        
        # Aggregation pipeline to get latest metrics per server
        pipeline = [
            {"$match": base_query},
            {"$sort": {"createdAt": -1}},
            {
                "$group": {
                    "_id": {"server": "$server", "environment": "$environment"},
                    "latest_metric": {"$first": "$$ROOT"},
                    "total_logs": {"$sum": 1}
                }
            },
            {"$replaceRoot": {"newRoot": "$latest_metric"}},
            {"$sort": {"environment": 1, "server": 1}}
        ]
        
        server_logs = list(server_col.aggregate(pipeline))
        logger.info(f"DEBUG: Found {len(server_logs)} server logs after aggregation")
        
        # Process metrics (same as before)
        metrics = []
        for log in server_logs:
            try:
                db_env = log.get('environment', '')
                env_mapping = {
                    "Dev": "Development",
                    "Stage": "Staging", 
                    "Prod": "Production",
                    "QA": "QA"
                }
                
                metric = {
                    "id": str(log.get("_id")),
                    "timestamp": log.get("timestamp"),
                    "environment": env_mapping.get(db_env, db_env),
                    "server": log.get("server"),
                    "app_name": log.get("server"),
                    "cpu_usage": float(log.get("cpu_usage", 0)),
                    "cpu_temp": float(log.get("cpu_temp", 0)),
                    "memory_usage": float(log.get("memory_usage", 0)),
                    "disk_utilization": float(log.get("disk_utilization", 0)),
                    "power_consumption": float(log.get("power_consumption", 0)),
                    "clock_speed": float(log.get("clock_speed", 0)),
                    "cache_miss_rate": float(log.get("cache_miss_rate", 0)),
                    "server_health": log.get("server_health"),
                    "ip_address": log.get("ip_address"),
                    "cpu_name": log.get("cpu_name"),
                    "path": log.get("path"),
                    "log_type": log.get("log_type"),
                    "createdAt": log.get("createdAt")
                }
                metrics.append(metric)
            except Exception as e:
                logger.error(f"ERROR: Processing server log {log.get('_id')}: {e}")
                continue
        
        logger.info(f"DEBUG: Returning {len(metrics)} server metrics")
        return metrics
        
    except Exception as e:
        logger.error(f"ERROR: Server metrics error: {e}")
        import traceback
        traceback.print_exc()
        return []

    
# Combined logs endpoint
@app.get("/api/combined_logs")
async def get_combined_logs(limit: int = 50, environment: Optional[str] = None, 
                           server: Optional[str] = None, app_name: Optional[str] = None):
    try:
        query = {}
        if environment and environment != "All":
            query["environment"] = environment
        if server and server != "All":
            query["server"] = server
        if app_name and app_name != "All":
            query["app_name"] = app_name

        app_logs = list(app_col.find(query).sort("createdAt", -1).limit(limit))
        
        formatted_logs = []
        for log in app_logs:
            formatted_log = {
                "log": {
                    "id": str(log.get("_id")),
                    "timestamp": log.get("timestamp", ""),
                    "level": log.get("level", "INFO"),
                    "message": log.get("message", ""),
                    "source": f"{log.get('environment', 'Unknown')}/{log.get('server', 'Unknown')}/{log.get('app_name', 'Unknown')}",
                    "environment": log.get("environment"),
                    "server": log.get("server"),
                    "app_name": log.get("app_name"),
                    "logger": log.get("logger"),
                    "thread": log.get("thread"),
                    "pid": log.get("pid"),
                    "exception_type": log.get("exception_type"),
                    "exception_message": log.get("exception_message"),
                    "stacktrace": log.get("stacktrace"),
                    "createdAt": log.get("createdAt")
                }
            }
            formatted_logs.append(formatted_log)
        
        return formatted_logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# App logs endpoint
@app.get("/api/app-logs")
async def get_app_logs(
    environment: Optional[str] = Query("All"),
    app_name: Optional[str] = Query("All"),
    search: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    limit: int = Query(100),
    page: int = Query(1)
):
    """Get application logs with proper debugging"""
    try:
        logger.info(f"DEBUG: App logs request - env: {environment}, app: {app_name}")
        
        # Build base query
        base_query = {}
        
        # Environment filter
        if environment and environment != "All":
            env_mapping = {
                "Development": "Dev",
                "Staging": "Stage",
                "Production": "Prod",
                "QA": "QA"
            }
            db_env = env_mapping.get(environment, environment)
            base_query['environment'] = db_env
            logger.info(f"DEBUG: Environment filter applied: {db_env}")
        
        # Application filter
        if app_name and app_name != "All":
            base_query['app_name'] = app_name
            logger.info(f"DEBUG: App filter applied: {app_name}")
        
        # Add search filter
        if search:
            base_query['$or'] = [
                {'message': {'$regex': search, '$options': 'i'}},
                {'logger': {'$regex': search, '$options': 'i'}}
            ]
        
        # Add level filter
        if level and level != "ALL":
            base_query['level'] = level
        
        logger.info(f"DEBUG: Final query: {base_query}")
        
        # Check total documents in collection
        total_docs = app_col.count_documents({})
        logger.info(f"DEBUG: Total documents in app collection: {total_docs}")
        
        # Check documents matching query
        matching_docs = app_col.count_documents(base_query)
        logger.info(f"DEBUG: Documents matching query: {matching_docs}")
        
        # Calculate skip for pagination
        skip = (page - 1) * limit
        
        # Get logs with pagination
        cursor = app_col.find(base_query).sort('createdAt', -1).skip(skip).limit(limit)
        logs = list(cursor)
        
        logger.info(f"DEBUG: Found {len(logs)} app logs")
        
        # Sample a few logs for debugging
        if logs:
            logger.info(f"DEBUG: Sample log: {logs[0]}")
        
        # Map database environment back to frontend format
        for log in logs:
            db_env = log.get('environment', '')
            env_mapping = {
                "Dev": "Development",
                "Stage": "Staging",
                "Prod": "Production", 
                "QA": "QA"
            }
            log['environment'] = env_mapping.get(db_env, db_env)
        
        serialized_logs = serialize_docs(logs)
        logger.info(f"DEBUG: Returning {len(serialized_logs)} serialized logs")
        
        return serialized_logs
        
    except Exception as e:
        logger.error(f"ERROR: App logs error: {e}")
        import traceback
        traceback.print_exc()
        return []


# Network metrics endpoint
@app.get("/api/network-metrics")
async def get_network_metrics(environment: Optional[str] = Query("All"), app_name: Optional[str] = Query("All")):
    """Get network metrics"""
    try:
        mock_logs = [
            {
                "id": "1",
                "timestamp": datetime.utcnow().isoformat(),
                "source_ip": "192.168.1.100",
                "destination_ip": "10.0.0.50", 
                "protocol": "HTTP",
                "port": 80,
                "bytes_transferred": 1024,
                "status": "Success",
                "response_time": 120,
                "environment": "Production",
                "app_name": "server1"
            },
            {
                "id": "2", 
                "timestamp": (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
                "source_ip": "192.168.1.101",
                "destination_ip": "10.0.0.51",
                "protocol": "HTTPS", 
                "port": 443,
                "bytes_transferred": 2048,
                "status": "Failed",
                "response_time": 5000,
                "environment": "Production",
                "app_name": "server2"
            }
        ]
        
        filtered_logs = mock_logs
        if environment != "All":
            filtered_logs = [log for log in filtered_logs if log['environment'] == environment]
        if app_name != "All":
            filtered_logs = [log for log in filtered_logs if log['app_name'] == app_name]
        
        total_requests = len(filtered_logs)
        successful_requests = len([log for log in filtered_logs if log['status'] == 'Success'])
        failed_requests = total_requests - successful_requests
        avg_response_time = sum(log['response_time'] for log in filtered_logs) / total_requests if total_requests > 0 else 0
        total_bandwidth = sum(log['bytes_transferred'] for log in filtered_logs)
        
        return {
            "logs": filtered_logs,
            "stats": {
                "totalRequests": total_requests,
                "successfulRequests": successful_requests,
                "failedRequests": failed_requests,
                "avgResponseTime": round(avg_response_time),
                "totalBandwidth": total_bandwidth
            }
        }
        
    except Exception as e:
        return {
            "logs": [],
            "stats": {
                "totalRequests": 0,
                "successfulRequests": 0,
                "failedRequests": 0,
                "avgResponseTime": 0,
                "totalBandwidth": 0
            }
        }
@app.get("/api/alerts")
async def get_alerts():
    """Get system alerts - compatibility endpoint"""
    try:
        # Get alerts from AlertManager collection
        active_alerts = list(alerts_col.find({"status": "firing"}).sort("startsAt", -1).limit(10))
        
        # Also get high usage alerts from server metrics
        high_usage_pipeline = [
            {"$sort": {"createdAt": -1}},
            {
                "$group": {
                    "_id": "$server",
                    "latest_log": {"$first": "$$ROOT"}
                }
            },
            {"$replaceRoot": {"newRoot": "$latest_log"}}
        ]
        
        cursor = server_col.aggregate(high_usage_pipeline)
        all_servers = list(cursor)
        
        alerts = []
        
        # Add AlertManager alerts
        for alert in active_alerts:
            alerts.append({
                "type": "error" if alert.get("labels", {}).get("severity") == "critical" else "warning",
                "title": alert.get("alertname", "Unknown Alert"),
                "message": alert.get("annotations", {}).get("summary", "No description available"),
                "server_name": alert.get("server_name"),
                "environment": alert.get("environment")
            })
        
        # Add server resource alerts
        for server in all_servers:
            try:
                cpu = float(server.get('cpu_usage', 0))
                memory = float(server.get('memory_usage', 0))
                disk = float(server.get('disk_utilization', 0))
                
                if cpu > 90 or memory > 90 or disk > 90:
                    alerts.append({
                        "type": "error",
                        "title": f"Critical Resource Usage - {server['server']}",
                        "message": f"CPU: {cpu}%, Memory: {memory}%, Disk: {disk}%"
                    })
                elif cpu > 80 or memory > 80 or disk > 80:
                    alerts.append({
                        "type": "warning", 
                        "title": f"High Resource Usage - {server['server']}",
                        "message": f"CPU: {cpu}%, Memory: {memory}%, Disk: {disk}%"
                    })
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing server {server.get('server', 'unknown')}: {e}")
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        return [
            {
                "type": "error",
                "title": "High CPU Usage",
                "message": "Server 1 CPU usage is above 90%"
            },
            {
                "type": "warning", 
                "title": "Memory Warning",
                "message": "Server 3 memory usage is above 80%"
            }
        ]

@app.get("/api/alertmanager/alerts")
async def get_alertmanager_alerts():
    """Get alerts from external AlertManager instance"""
    try:
        resp = requests.get("http://localhost:9093/api/v2/alerts", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"Could not fetch from external AlertManager: {e}")
        return {"error": str(e)}

# Alerts endpoint
@app.post("/api/alerts/webhook")
async def receive_alertmanager_webhook(alert_data: AlertManagerWebhook = Body(...)):
    logger.info(f"Received Alertmanager webhook: Status={alert_data.status}, Num_Alerts={len(alert_data.alerts)}")
    
    processed_count = 0
    failed_count = 0
    errors = []
    
    for i, alert in enumerate(alert_data.alerts):
        try:
            logger.info(f"Processing alert {i+1}: {alert.labels.get('alertname')} - {alert.status}")
            
            # FIXED: Create unique alert_id using more distinguishing fields
            alertname = alert.labels.get('alertname', 'unknown')
            instance = alert.labels.get('instance', 'unknown')
            server_name = alert.labels.get('server_name', alert.labels.get('server', 'unknown'))
            environment = alert.labels.get('environment', alert.labels.get('env', 'unknown'))
            
            # Create a more unique alert_id
            alert_id = f"{alertname}-{server_name}-{environment}-{instance}-{alert.startsAt}"
            
            logger.info(f"Generated alert_id: {alert_id}")
            
            # Safe datetime parsing
            def safe_parse_datetime(date_str):
                if not date_str:
                    return None
                try:
                    if date_str.endswith('Z'):
                        date_str = date_str[:-1] + '+00:00'
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                except Exception as e:
                    logger.error(f"Failed to parse datetime {date_str}: {e}")
                    return datetime.utcnow()
            
            alert_doc = {
                "alert_id": alert_id,
                "status": alert.status,
                "labels": alert.labels,
                "annotations": alert.annotations,
                "startsAt": safe_parse_datetime(alert.startsAt),
                "endsAt": safe_parse_datetime(alert.endsAt) if alert.endsAt else None,
                "generatorURL": alert.generatorURL,
                "receivedAt": datetime.utcnow(),
                "processed": False,
                # Add these for easier querying
                "alertname": alertname,
                "server_name": server_name,
                "environment": environment,
                "instance": instance
            }
            
            logger.info(f"Attempting to store alert: {alertname} for {server_name} in {environment}")
            
            result = alerts_col.update_one(
                {"alert_id": alert_id},
                {"$set": alert_doc},
                upsert=True
            )
            
            if result.upserted_id:
                logger.info(f"✅ INSERTED NEW alert: {alert_id}")
                processed_count += 1
            elif result.modified_count > 0:
                logger.info(f"✅ UPDATED existing alert: {alert_id}")
                processed_count += 1
            else:
                logger.warning(f"⚠️ No changes made to alert: {alert_id} (probably identical)")
                processed_count += 1  # Still count as processed
                
        except Exception as e:
            failed_count += 1
            error_msg = f"Failed to process alert {i+1} ({alert.labels.get('alertname', 'unknown')}): {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
    
    response = {
        "status": "success" if processed_count > 0 else "failed",
        "processed_alerts": processed_count,
        "failed_alerts": failed_count,
        "total_alerts": len(alert_data.alerts)
    }
    
    if errors:
        response["errors"] = errors
    
    logger.info(f"Webhook processing complete: {response}")
    return response


# Add this debugging endpoint to check what's actually in your DB
@app.get("/api/debug/alerts")
async def debug_alerts():
    try:
        # Get total count
        total_alerts = alerts_col.count_documents({})
        
        # Get recent alerts
        recent_alerts = list(alerts_col.find({}).sort("receivedAt", -1).limit(20))
        
        # Get unique alert names
        unique_alertnames = alerts_col.distinct("alertname")
        
        # Get unique servers
        unique_servers = alerts_col.distinct("server_name")
        
        # Get unique environments  
        unique_environments = alerts_col.distinct("environment")
        
        # Convert ObjectIds and dates for JSON
        for alert in recent_alerts:
            alert["_id"] = str(alert["_id"])
            if "startsAt" in alert and alert["startsAt"]:
                alert["startsAt"] = alert["startsAt"].isoformat()
            if "endsAt" in alert and alert["endsAt"]:
                alert["endsAt"] = alert["endsAt"].isoformat()
            if "receivedAt" in alert and alert["receivedAt"]:
                alert["receivedAt"] = alert["receivedAt"].isoformat()
        
        return {
            "total_alerts_in_db": total_alerts,
            "unique_alertnames": unique_alertnames,
            "unique_servers": unique_servers,
            "unique_environments": unique_environments,
            "recent_alerts": recent_alerts,
            "collection_name": alerts_col.name,
            "database_name": alerts_col.database.name
        }
    except Exception as e:
        logger.error(f"Debug alerts error: {e}")
        return {"error": str(e)}


# Also fix your get_active_alerts to handle the new structure
@app.get("/api/alerts/active")
async def get_active_alerts(
    environment: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50
):
    try:
        query = {"status": "firing"}
        
        if environment:
            query["environment"] = environment
        if severity:
            query["labels.severity"] = severity
            
        logger.info(f"Querying active alerts with: {query}")
        
        alerts = list(alerts_col.find(query).sort("startsAt", -1).limit(limit))
        
        logger.info(f"Found {len(alerts)} active alerts")
        
        # Convert to frontend format
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "id": str(alert.get("_id")),
                "alertname": alert.get("alertname", alert["labels"].get("alertname")),
                "severity": alert["labels"].get("severity", "warning"),
                "summary": alert["annotations"].get("summary"),
                "description": alert["annotations"].get("description"),
                "server_name": alert.get("server_name", alert["labels"].get("server_name", alert["labels"].get("instance"))),
                "environment": alert.get("environment", alert["labels"].get("environment", alert["labels"].get("env"))),
                "issue_type": alert["labels"].get("issue_type"),
                "startsAt": alert["startsAt"].isoformat() if alert.get("startsAt") else None,
                "generatorURL": alert.get("generatorURL"),
                "status": alert["status"]
            })
        
        logger.info(f"Returning {len(formatted_alerts)} formatted alerts")
        return formatted_alerts
        
    except Exception as e:
        logger.error(f"Error fetching active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/alerts/history")
async def get_alert_history(
    days: int = 7,
    environment: Optional[str] = None,
    severity: Optional[str] = None
):
    try:
        query = {
            "startsAt": {
                "$gte": datetime.utcnow() - timedelta(days=days)
            }
        }
        
        if environment:
            query["labels.env"] = environment
        if severity:
            query["labels.severity"] = severity
            
        alerts = list(alerts_col.find(query).sort("startsAt", -1).limit(100))
        
        # Convert to frontend format
        formatted_alerts = []
        for alert in alerts:
            formatted_alerts.append({
                "id": str(alert.get("_id")),
                "alertname": alert["labels"].get("alertname"),
                "severity": alert["labels"].get("severity", "warning"),
                "summary": alert["annotations"].get("summary"),
                "description": alert["annotations"].get("description"),
                "server_name": alert["labels"].get("instance"),
                "environment": alert["labels"].get("env"),
                "issue_type": alert["labels"].get("issue_type"),
                "startsAt": alert["startsAt"].isoformat(),
                "endsAt": alert.get("endsAt", "").isoformat() if alert.get("endsAt") else None,
                "status": alert["status"],
                "generatorURL": alert.get("generatorURL")
            })
        
        return formatted_alerts
    except Exception as e:
        logger.error(f"Error fetching alert history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/api/chat/message")
async def send_chat_message(request: ChatMessage):
    try:
        message = request.message.lower()
        context = request.context or {}
        
        infrastructure_data = {}
        
        # Query recent server metrics if relevant - FIXED VERSION
        if any(keyword in message for keyword in ['server', 'cpu', 'memory', 'performance', 'metrics', 'dev', 'server1']):
            try:
                # Use your actual server collection and proper field names
                recent_metrics = list(server_col.find({
                    "server": {"$exists": True, "$ne": None},
                    "environment": {"$exists": True, "$ne": None}
                }).sort("createdAt", -1).limit(10))
                
                infrastructure_data['recent_metrics'] = []
                for metric in recent_metrics:
                    # Map database environment to frontend format
                    env_mapping = {
                        "Dev": "Development",
                        "Prod": "Production", 
                        "Stage": "Staging",
                        "QA": "QA"
                    }
                    
                    infrastructure_data['recent_metrics'].append({
                        "server": metric.get("server", "Unknown"),
                        "environment": env_mapping.get(metric.get("environment"), metric.get("environment", "Unknown")),
                        "cpu_usage": metric.get("cpu_usage", 0),
                        "memory_usage": metric.get("memory_usage", 0),
                        "disk_utilization": metric.get("disk_utilization", 0),
                        "cpu_temp": metric.get("cpu_temp", 0),
                        "server_health": metric.get("server_health", "Unknown"),
                        "ip_address": metric.get("ip_address", "N/A"),
                        "timestamp": metric.get("createdAt", "").isoformat() if metric.get("createdAt") else "N/A"
                    })
                
                logger.info(f"Found {len(infrastructure_data['recent_metrics'])} server metrics for chat")
                
            except Exception as e:
                logger.warning(f"Could not fetch server metrics: {e}")
        
        # Query recent logs if relevant - FIXED VERSION
        if any(keyword in message for keyword in ['log', 'error', 'issue', 'problem', 'alert']):
            try:
                recent_logs = list(app_col.find({
                    "level": {"$in": ["ERROR", "WARN", "CRITICAL"]},
                    "server": {"$exists": True},
                    "environment": {"$exists": True}
                }).sort("createdAt", -1).limit(5))
                
                infrastructure_data['recent_issues'] = []
                for log in recent_logs:
                    infrastructure_data['recent_issues'].append({
                        "level": log.get("level"),
                        "message": log.get("message", "")[:100],
                        "server": log.get("server", "Unknown"),
                        "environment": log.get("environment", "Unknown"),
                        "timestamp": log.get("createdAt", "").isoformat() if log.get("createdAt") else "N/A"
                    })
                
                logger.info(f"Found {len(infrastructure_data['recent_issues'])} recent issues for chat")
                
            except Exception as e:
                logger.warning(f"Could not fetch logs: {e}")
        
        # Query predictive maintenance flags if relevant
        if any(keyword in message for keyword in ['predict', 'maintenance', 'failure', 'risk']):
            try:
                predictive_col = db['predictive_maintenance_flags']
                predictive_flags = list(predictive_col.find({}).sort("prediction_timestamp", -1).limit(5))
                
                infrastructure_data['predictive_alerts'] = []
                for flag in predictive_flags:
                    infrastructure_data['predictive_alerts'].append({
                        "server": flag.get("server_name", "Unknown"),
                        "environment": flag.get("environment", "Unknown"),
                        "predicted_issue": flag.get("predicted_issue", "")[:100],
                        "confidence": flag.get("confidence", "Unknown")
                    })
                
                logger.info(f"Found {len(infrastructure_data['predictive_alerts'])} predictive alerts for chat")
                
            except Exception as e:
                logger.warning(f"Could not fetch predictive flags: {e}")
        
        # Create enhanced prompt with real data
        data_context = ""
        if infrastructure_data:
            data_context = f"\nCurrent Infrastructure Data:\n{json.dumps(infrastructure_data, indent=2)}\n"
        
        # Enhanced prompt that specifically looks for server queries
        prompt = f"""
        You are an intelligent infrastructure monitoring assistant with access to real-time data.
        
        Available environments: Development (Dev), Staging (Stage), Production (Prod), QA
        Available servers: server1, server2, server3, server4
        
        {data_context}
        
        User question: {request.message}
        
        IMPORTANT: If the user asks about a specific server (like "server1 of dev" or "dev server1"), 
        look through the recent_metrics data to find that exact server and environment combination.
        
        Provide specific, actionable responses based on the available data. If you see concerning metrics or errors, highlight them and suggest next steps.
        Keep responses concise but informative. Focus on infrastructure monitoring, server health, and system performance.
        
        If no data is available for the requested server, explain that monitoring data is not currently available for that server.
        """
        
        response = model.generate_content(prompt)
        response_text = clean_gemini_response(response.text)
        
        # Store chat history
        chat_record = {
            "user_message": request.message,
            "bot_response": response_text,
            "context": {**context, "infrastructure_data": infrastructure_data},
            "timestamp": datetime.utcnow(),
            "session_id": context.get("session_id", "default")
        }
        
        chat_col.insert_one(chat_record)
        
        return {
            "response": response_text,
            "context": {
                "model": "gemini-1.5-flash",
                "infrastructure_data": infrastructure_data,
                "timestamp": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {str(e)}")
        return {
            "response": "I'm having trouble accessing the infrastructure data right now. Please try again in a moment.",
            "context": {"error": True},
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/chat/history")
async def get_chat_history(limit: int = 50, session_id: str = "default"):
    try:
        # Query chat history with proper field names
        history = list(chat_col.find(
            {"session_id": session_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(limit))
        
        logger.info(f"Found {len(history)} chat history records")
        
        formatted_history = []
        for chat in reversed(history):  # Reverse to show oldest first
            # Add user message
            formatted_history.append({
                "id": f"user_{int(chat['timestamp'].timestamp() * 1000)}",
                "type": "user",
                "content": chat["user_message"],
                "timestamp": chat["timestamp"].isoformat()
            })
            
            # Add bot response with context
            formatted_history.append({
                "id": f"bot_{int(chat['timestamp'].timestamp() * 1000)}",
                "type": "bot", 
                "content": chat["bot_response"],
                "context": chat.get("context", {}),
                "timestamp": chat["timestamp"].isoformat(),
                "has_context": bool(chat.get("context", {}).get("infrastructure_data"))
            })
        
        return formatted_history
        
    except Exception as e:
        logger.error(f"Chat history error: {str(e)}")
        return []
    
@app.get("/api/chat/context/{message_id}")
async def get_chat_context(message_id: str):
    """Get detailed context for a specific chat message"""
    try:
        # Extract timestamp from message_id
        timestamp_ms = int(message_id.split('_')[1])
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
        
        # Find the chat record
        chat_record = chat_col.find_one({
            "timestamp": {
                "$gte": timestamp - timedelta(seconds=1),
                "$lte": timestamp + timedelta(seconds=1)
            }
        })
        
        if not chat_record:
            return {"error": "Context not found"}
        
        return {
            "context": chat_record.get("context", {}),
            "user_message": chat_record.get("user_message"),
            "bot_response": chat_record.get("bot_response"),
            "timestamp": chat_record.get("timestamp").isoformat()
        }
        
    except Exception as e:
        logger.error(f"Context retrieval error: {str(e)}")
        return {"error": "Failed to retrieve context"}

# Log analysis endpoint
@app.get("/api/log_analysis/{log_id}")
def get_log_analysis(log_id: str):
    log_id = log_id.strip()
    
    if not log_id:
        return JSONResponse(status_code=400, content={"error": "Invalid log ID"})
    
    logger.info(f"Looking for original_log_id: '{log_id}'")
    
    try:
        doc = None
        
        # Try multiple search strategies
        doc = analysis_collection.find_one({"original_log_id": log_id})
        
        if not doc and ObjectId.is_valid(log_id):
            doc = analysis_collection.find_one({"original_log_id": ObjectId(log_id)})
            
        if not doc and ObjectId.is_valid(log_id):
            doc = analysis_collection.find_one({"_id": ObjectId(log_id)})
            
        if not doc:
            return JSONResponse(status_code=404, content={"error": "Analysis not found"})
            
        # Convert ObjectIds to strings for JSON serialization
        doc["_id"] = str(doc["_id"])
        if "original_log_id" in doc:
            doc["original_log_id"] = str(doc["original_log_id"])
            
        return doc
        
    except Exception as e:
        logger.error(f"Error in get_log_analysis: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal server error"})

# Debug endpoints
@app.get("/api/debug/collection-info")
async def debug_collection_info():
    """Debug endpoint to inspect collection structure"""
    try:
        sample_doc = server_col.find_one()
        
        environments = list(server_col.distinct('environment'))
        servers = list(server_col.distinct('server'))
        log_types = list(server_col.distinct('log_type'))
        
        return {
            "sample_document_keys": list(sample_doc.keys()) if sample_doc else [],
            "total_documents": server_col.count_documents({}),
            "distinct_environments": environments,
            "distinct_servers": servers,
            "distinct_log_types": log_types,
            "sample_document": serialize_doc(sample_doc) if sample_doc else None
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/app_deployment/{app_name}")
async def debug_app_deployment(app_name: str, environment: Optional[str] = None):
    try:
        pipeline = [{"$match": {"app_name": app_name}}]
        
        if environment:
            pipeline[0]["$match"]["environment"] = environment
            
        pipeline.append({
            "$group": {
                "_id": {"server": "$server", "environment": "$environment"},
                "log_count": {"$sum": 1}
            }
        })
        
        result = list(app_col.aggregate(pipeline))
        return {
            "app": app_name,
            "environment": environment,
            "deployed_on": result
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug/all_servers")
async def debug_all_servers():
    try:
        pipeline = [
            {"$match": {
                "server": {"$ne": None, "$ne": "", "$exists": True},
                "environment": {"$ne": None, "$ne": "", "$exists": True}
            }},
            {"$group": {
                "_id": {"server": "$server", "environment": "$environment"},
                "last_seen": {"$max": "$createdAt"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.environment": 1, "_id.server": 1}}
        ]
        
        all_servers = list(cpu_col.aggregate(pipeline))
        return {
            "total_unique_servers": len(all_servers),
            "servers": all_servers
        }
    except Exception as e:
        return {"error": str(e)}

# Static file serving for production
frontend_build_path = "../frontend/build"
static_path = "../frontend/build/static"

if os.path.exists(frontend_build_path) and os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
    templates = Jinja2Templates(directory=frontend_build_path)
    
    @app.get("/{rest_of_path:path}")
    async def serve_react_app(request: Request, rest_of_path: str):
        return templates.TemplateResponse("index.html", {"request": request})
    
    logger.info("✅ Static file serving enabled")
else:
    logger.info("⚠️  Frontend build directory not found. Running in API-only mode.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

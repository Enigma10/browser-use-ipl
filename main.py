"""IPL-specific browser automation tools."""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional
import os
import json
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.prompts import AgentMessagePrompt, SystemPrompt
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("browser_use")

# Global variables
browser: Optional[Browser] = None
browser_context: Optional[BrowserContext] = None
message_manager: Optional[MessageManager] = None
CACHE_FILE = os.path.join(os.path.dirname(__file__), 'ipl_cache.json')
BROWSER_INITIALIZED_BY_FETCH = False  # Track if browser was initialized by fetch_fresh_data

# Cache management functions
def initialize_cache():
    """Initialize the cache file if it doesn't exist."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                existing_cache = json.load(f)
                return existing_cache
    except Exception as e:
        logger.error(f"Error reading existing cache: {str(e)}")
    
    # Create new cache if reading failed or file doesn't exist
    default_cache = {
        'schedule': {
            'lastUpdated': None,
            'matches': []
        },
        'pointsTable': {
            'lastUpdated': None,
            'teams': []
        },
        'lastChecked': {
            'schedule': None,
            'pointsTable': None
        }
    }
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        save_cache(default_cache)
        logger.info(f"Initialized new cache at {CACHE_FILE}")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {str(e)}")
    
    return default_cache

def load_cache():
    """Load the cache from file."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                logger.debug(f"Successfully loaded cache from {CACHE_FILE}")
                return cache_data
        else:
            logger.info(f"Cache file not found, initializing new cache")
            return initialize_cache()
    except Exception as e:
        logger.error(f"Error loading cache: {str(e)}")
        return initialize_cache()

def save_cache(data):
    """Save data to cache file."""
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Successfully saved cache to {CACHE_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")
        return False

def is_checked_recently(cache: Dict, data_type: str, minutes: int = 30) -> bool:
    """Check if data type was checked recently to avoid repeated browser initialization."""
    if not cache.get('lastChecked') or not cache['lastChecked'].get(data_type):
        logger.debug(f"{data_type} has never been checked")
        return False
    
    last_check_str = cache['lastChecked'][data_type]
    try:
        last_check = datetime.fromisoformat(last_check_str)
        time_since_last_check = datetime.now() - last_check
        recently_checked = time_since_last_check < timedelta(minutes=minutes)
        
        logger.debug(f"{data_type} last checked at {last_check_str}")
        logger.debug(f"Time since last check: {time_since_last_check}")
        logger.debug(f"Recently checked: {recently_checked}")
        
        return recently_checked
    except Exception as e:
        logger.error(f"Error checking last check time: {str(e)}")
        return False

def is_too_old(timestamp: str, days=0, minutes=0) -> bool:
    """Check if the timestamp is too old."""
    if not timestamp:
        logger.debug("No timestamp provided, considering data as too old")
        return True
    
    try:
        last_update = datetime.fromisoformat(timestamp)
        max_age = timedelta(days=days, minutes=minutes)
        too_old = datetime.now() - last_update > max_age
        
        logger.debug(f"Last update: {last_update}")
        logger.debug(f"Current time: {datetime.now()}")
        logger.debug(f"Max age: {max_age}")
        logger.debug(f"Data is too old: {too_old}")
        
        return too_old
    except Exception as e:
        logger.error(f"Error checking if timestamp is too old: {str(e)}")
        return True

def is_updated_today(timestamp: str) -> bool:
    """Check if the timestamp is from today."""
    if not timestamp:
        logger.debug("No timestamp provided, data not updated today")
        return False
    
    try:
        last_update = datetime.fromisoformat(timestamp)
        current_date = datetime.now().date()
        is_today = last_update.date() == current_date
        
        logger.debug(f"Last update date: {last_update.date()}")
        logger.debug(f"Current date: {current_date}")
        logger.debug(f"Updated today: {is_today}")
        
        return is_today
    except Exception as e:
        logger.error(f"Error checking if updated today: {str(e)}")
        return False

def update_last_checked(data_type: str):
    """Update the last checked timestamp for a specific data type."""
    try:
        cache = load_cache()
        if not cache.get('lastChecked'):
            cache['lastChecked'] = {}
        
        cache['lastChecked'][data_type] = datetime.now().isoformat()
        save_cache(cache)
        logger.debug(f"Updated last checked time for {data_type}")
    except Exception as e:
        logger.error(f"Error updating last checked timestamp: {str(e)}")

def get_live_matches_from_schedule(schedule_data: Dict) -> list:
    """Get live matches from schedule data."""
    if not schedule_data or not schedule_data.get('matches'):
        return []
    
    current_time = datetime.now()
    live_matches = []
    
    for match in schedule_data['matches']:
        match_time = datetime.fromisoformat(match.get('date', ''))
        # Consider a match potentially live if it's today
        if match_time.date() == current_time.date():
            live_matches.append(match)
    
    return live_matches

def parse_points_table(content: str) -> dict:
    """Parse points table content into structured data."""
    # This will parse the HTML content from cricbuzz into structured data
    # For now returning raw content, we'll implement parsing later
    return {
        "content": content,
        "parsed": False  # Indicator that we need to implement parsing
    }

def parse_schedule(content: str) -> dict:
    """Parse schedule content into structured data."""
    # This will parse the HTML content from cricbuzz into structured data
    # For now returning raw content, we'll implement parsing later
    return {
        "content": content,
        "parsed": False  # Indicator that we need to implement parsing
    }

async def fetch_fresh_data(data_type: str) -> Dict:
    """Fetch fresh data from cricbuzz based on data type."""
    global browser, browser_context, BROWSER_INITIALIZED_BY_FETCH
    
    logger.info(f"Fetching fresh data for {data_type}")
    browser_initialized_locally = False
    
    try:
        if not browser_context:
            logger.debug("Initializing browser for fetch_fresh_data")
            await initialize_browser(headless=True)
            browser_initialized_locally = True
            BROWSER_INITIALIZED_BY_FETCH = True

        if data_type == 'schedule':
            await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/matches")
            await wait(2)
            content = await inspect_page()
            if not content:
                raise Exception("Failed to get schedule content")
            return {
                'lastUpdated': datetime.now().isoformat(),
                'matches': parse_schedule(content)
            }
        
        elif data_type == 'pointsTable':
            await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/points-table")
            await wait(2)
            content = await inspect_page()
            if not content:
                raise Exception("Failed to get points table content")
            return {
                'lastUpdated': datetime.now().isoformat(),
                'teams': parse_points_table(content)
            }
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    except Exception as e:
        logger.error(f"Error fetching fresh data for {data_type}: {str(e)}")
        return None
    finally:
        # Only close the browser if we opened it in this function
        if browser_initialized_locally and BROWSER_INITIALIZED_BY_FETCH:
            logger.debug("Closing browser that was initialized by fetch_fresh_data")
            await close_browser()
            BROWSER_INITIALIZED_BY_FETCH = False

def update_cache(data_type: str, new_data: Dict):
    """Update specific data type in cache."""
    try:
        cache = load_cache()
        cache[data_type] = new_data
        save_cache(cache)
        logger.info(f"Updated cache for {data_type}")
    except Exception as e:
        logger.error(f"Error updating cache for {data_type}: {str(e)}")

# Essential browser functions for IPL
@mcp.tool()
async def initialize_browser(headless: bool = False, task: str = "") -> str:
    """Initialize browser for IPL operations."""
    global browser, browser_context
    
    if browser:
        logger.debug("Browser already initialized, closing it first")
        await close_browser()
    
    logger.info(f"Initializing browser for task: {task}")
    config = BrowserConfig(headless=headless)
    browser = Browser(config=config)
    browser_context = BrowserContext(browser=browser)
    
    return f"Browser initialized for task: {task}"

@mcp.tool()
async def close_browser() -> str:
    """Close the browser instance."""
    global browser, browser_context, BROWSER_INITIALIZED_BY_FETCH
    
    logger.info("Closing browser")
    
    if browser_context:
        await browser_context.close()
        browser_context = None
    
    if browser:
        await browser.close()
        browser = None
    
    BROWSER_INITIALIZED_BY_FETCH = False
    return "Browser closed successfully"

@mcp.tool()
async def go_to_url(url: str) -> str:
    """Navigate to a specific URL."""
    logger.info(f"Navigating to {url}")
    page = await browser_context.get_current_page()
    await page.goto(url)
    await page.wait_for_load_state()
    return f"Navigated to {url}"

@mcp.tool()
async def click_element(index: int) -> str:
    """Click an element on the page."""
    if index not in await browser_context.get_selector_map():
        raise Exception(f"Element with index {index} not found")
    
    element_node = await browser_context.get_dom_element_by_index(index)
    await browser_context._click_element_node(element_node)
    logger.info(f"Clicked element at index {index}")
    return f"Clicked element at index {index}"

@mcp.tool()
async def wait(seconds: int = 3) -> str:
    """Wait for specified seconds."""
    logger.debug(f"Waiting for {seconds} seconds")
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds"

@mcp.tool()
async def inspect_page() -> str:
    """
    Lists interactive elements and extracts content from the current page.
    Returns:
        str: A formatted string that lists all interactive elements (if any) along with the content.
    """
    logger.debug("Inspecting page")
    # Get the current state to inspect interactive elements
    state = await browser_context.get_state()
    prompt_message = AgentMessagePrompt(
        state,
        include_attributes=["type", "role", "placeholder", "aria-label", "title"],
    ).get_user_message(use_vision=False)
    return prompt_message.content

@mcp.tool()
async def scroll_down(amount: int = None) -> str:
    """Scroll down the page."""
    page = await browser_context.get_current_page()
    if amount:
        await page.evaluate(f"window.scrollBy(0, {amount})")
        logger.debug(f"Scrolled down by {amount} pixels")
    else:
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
        logger.debug("Scrolled down by one screen height")
    return "Scrolled down"

@mcp.tool()
async def scroll_up(amount: int = None) -> str:
    """Scroll up the page."""
    page = await browser_context.get_current_page()
    if amount:
        await page.evaluate(f"window.scrollBy(0, -{amount})")
        logger.debug(f"Scrolled up by {amount} pixels")
    else:
        await page.evaluate("window.scrollBy(0, -window.innerHeight)")
        logger.debug("Scrolled up by one screen height")
    return "Scrolled up"

# IPL-specific functions
@mcp.tool()
async def get_current_matches() -> Dict[str, Any]:
    """Get list of all current IPL matches."""
    try:
        logger.info("Getting current IPL matches")
        if not browser_context:
            await initialize_browser(headless=False, task="Get IPL Matches")
        
        await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/matches")
        await wait(2)
        
        matches_content = await inspect_page()
        return {"status": "success", "matches": matches_content}
    except Exception as e:
        logger.error(f"Error getting matches: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_match_details(match_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific IPL match."""
    try:
        logger.info(f"Getting details for match ID: {match_id}")
        if not browser_context:
            await initialize_browser(headless=False, task="Get Match Details")
        
        await go_to_url(f"https://www.cricbuzz.com/live-cricket-scores/{match_id}")
        await wait(2)
        
        match_content = await inspect_page()
        return {"status": "success", "details": match_content}
    except Exception as e:
        logger.error(f"Error getting match details: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_points_table() -> Dict[str, Any]:
    """Get current IPL points table."""
    try:
        logger.info("Getting IPL points table")
        
        # First check cache
        cache = load_cache()
        update_last_checked('pointsTable')
        
        # Decide whether to use cache or fetch fresh data
        need_refresh = False
        
        # Check if we've already checked recently (to avoid unnecessary browser init)
        recently_checked = is_checked_recently(cache, 'pointsTable', minutes=30)
        
        if recently_checked:
            logger.info("Points table checked recently, using cached data")
            return {"status": "success", "points_table": cache['pointsTable']['teams']}
        
        # If not checked recently, determine if we need fresh data
        if not cache.get('pointsTable'):
            logger.debug("Need refresh: No pointsTable in cache")
            need_refresh = True
        elif not cache['pointsTable'].get('lastUpdated'):
            logger.debug("Need refresh: No lastUpdated timestamp")
            need_refresh = True
        elif not is_updated_today(cache['pointsTable']['lastUpdated']):
            logger.debug(f"Need refresh: Not updated today. Last update: {cache['pointsTable']['lastUpdated']}")
            need_refresh = True
        else:
            logger.info("Using cached points table data from today")
        
        if need_refresh:
            # Get fresh data
            logger.info("Fetching fresh points table data")
            points_data = await fetch_fresh_data('pointsTable')
            if points_data:
                update_cache('pointsTable', points_data)
                return {"status": "success", "points_table": points_data['teams']}
            else:
                # If fetch fails, use whatever we have in cache
                if cache.get('pointsTable') and cache['pointsTable'].get('teams'):
                    logger.warning("Fetch failed, using cached data despite being old")
                    return {"status": "success", "points_table": cache['pointsTable']['teams']}
                return {"status": "error", "message": "Failed to fetch points table and no cache available"}
        else:
            # Return cached data
            return {"status": "success", "points_table": cache['pointsTable']['teams']}
            
    except Exception as e:
        logger.error(f"Error getting points table: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_stats() -> Dict[str, Any]:
    """Get IPL statistics (batting and bowling)."""
    try:
        logger.info("Getting IPL statistics")
        if not browser_context:
            await initialize_browser(headless=False, task="Get IPL Stats")
        
        await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/stats")
        await wait(2)
        
        stats_content = await inspect_page()
        
        try:
            await click_element(37)  # Bowling stats tab
            await wait(1)
            bowling_stats = await inspect_page()
            stats_content += "\n\nBowling Stats:\n" + bowling_stats
        except Exception as e:
            logger.warning(f"Could not load bowling stats: {str(e)}")
        
        return {"status": "success", "stats": stats_content}
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_team_schedule(team_name: str) -> Dict[str, Any]:
    """Get schedule for a specific IPL team."""
    try:
        logger.info(f"Getting schedule for team: {team_name}")
        
        # First check if we have the full schedule in cache
        cache = load_cache()
        update_last_checked('schedule')
        
        # Check if we've already checked recently
        recently_checked = is_checked_recently(cache, 'schedule', minutes=30)
        
        if recently_checked and cache.get('schedule') and cache['schedule'].get('matches'):
            logger.info("Using cached schedule to filter team matches")
            return filter_team_matches(cache['schedule']['matches'], team_name)
        
        # If cache is not recent or empty, fetch fresh data
        schedule_data = await fetch_fresh_data('schedule')
        if schedule_data:
            update_cache('schedule', schedule_data)
            return filter_team_matches(schedule_data['matches'], team_name)
        
        # If fetch fails but we have cache, use it
        if cache.get('schedule') and cache['schedule'].get('matches'):
            logger.warning("Fetch failed, using cached data for team schedule")
            return filter_team_matches(cache['schedule']['matches'], team_name)
        
        return {
            "status": "error",
            "message": "Failed to fetch team schedule and no cache available"
        }
        
    except Exception as e:
        logger.error(f"Error getting team schedule: {str(e)}")
        return {"status": "error", "message": str(e)}

def filter_team_matches(matches_data: Dict, team_name: str) -> Dict[str, Any]:
    """Filter matches for a specific team from the schedule data."""
    try:
        team_matches = []
        if not matches_data or not matches_data.get('content'):
            return {
                "status": "error",
                "message": "Invalid matches data format"
            }
        
        content = matches_data['content']
        team_name_lower = team_name.lower()
        
        # Split content into lines and process each match
        for line in content.split('\n'):
            if team_name_lower in line.lower():
                team_matches.append(line.strip())
        
        return {
            "status": "success",
            "team": team_name,
            "matches": team_matches,
            "match_count": len(team_matches)
        }
        
    except Exception as e:
        logger.error(f"Error filtering team matches: {str(e)}")
        return {
            "status": "error",
            "message": f"Error processing team matches: {str(e)}"
        }

@mcp.tool()
async def get_ipl_schedule() -> Dict[str, Any]:
    """Get full IPL schedule."""
    try:
        logger.info("Getting IPL schedule")
        
        # First check cache
        cache = load_cache()
        update_last_checked('schedule')
        
        # Check if we've already checked recently (to avoid unnecessary browser init)
        recently_checked = is_checked_recently(cache, 'schedule', minutes=30)
        
        if recently_checked:
            logger.info("Schedule checked recently, using cached data")
            return {"status": "success", "schedule": cache['schedule']['matches']}
        
        # Decide whether to use cache or fetch fresh data
        need_refresh = False
        
        if not cache.get('schedule'):
            logger.debug("Need refresh: No schedule in cache")
            need_refresh = True
        elif not cache['schedule'].get('lastUpdated'):
            logger.debug("Need refresh: No lastUpdated timestamp")
            need_refresh = True
        elif is_too_old(cache['schedule']['lastUpdated'], days=7):
            logger.debug(f"Need refresh: Data is older than 7 days. Last update: {cache['schedule']['lastUpdated']}")
            need_refresh = True
        else:
            logger.info("Using cached schedule data (less than 7 days old)")
        
        if need_refresh:
            # Get fresh data
            logger.info("Fetching fresh schedule data")
            schedule_data = await fetch_fresh_data('schedule')
            if schedule_data:
                update_cache('schedule', schedule_data)
                return {"status": "success", "schedule": schedule_data['matches']}
            else:
                # If fetch fails, use whatever we have in cache
                if cache.get('schedule') and cache['schedule'].get('matches'):
                    logger.warning("Fetch failed, using cached data despite being old")
                    return {"status": "success", "schedule": cache['schedule']['matches']}
                return {"status": "error", "message": "Failed to fetch schedule and no cache available"}
        else:
            # Return cached data
            return {"status": "success", "schedule": cache['schedule']['matches']}
            
    except Exception as e:
        logger.error(f"Error getting schedule: {str(e)}")
        return {"status": "error", "message": str(e)}

def main():
    """Run the MCP server for IPL information."""
    logger.info("Starting MCP server for IPL information")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
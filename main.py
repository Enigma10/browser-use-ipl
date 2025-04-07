"""IPL-specific browser automation tools."""

import asyncio
import logging
import sys
from typing import Dict, Any, Optional

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

# Essential browser functions for IPL
@mcp.tool()
async def initialize_browser(headless: bool = False, task: str = "") -> str:
    """Initialize browser for IPL operations."""
    global browser, browser_context
    
    if browser:
        await close_browser()
    
    config = BrowserConfig(headless=headless)
    browser = Browser(config=config)
    browser_context = BrowserContext(browser=browser)
    
    return f"Browser initialized for task: {task}"

@mcp.tool()
async def close_browser() -> str:
    """Close the browser instance."""
    global browser, browser_context
    
    if browser_context:
        await browser_context.close()
        browser_context = None
    
    if browser:
        await browser.close()
        browser = None
    
    return "Browser closed successfully"

@mcp.tool()
async def go_to_url(url: str) -> str:
    """Navigate to a specific URL."""
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
    return f"Clicked element at index {index}"

@mcp.tool()
async def wait(seconds: int = 3) -> str:
    """Wait for specified seconds."""
    await asyncio.sleep(seconds)
    return f"Waited for {seconds} seconds"

@mcp.tool()
async def inspect_page() -> str:
    """Get page content and interactive elements."""
    page = await browser_context.get_current_page()
    content = await page.content()
    return content

@mcp.tool()
async def scroll_down(amount: int = None) -> str:
    """Scroll down the page."""
    page = await browser_context.get_current_page()
    if amount:
        await page.evaluate(f"window.scrollBy(0, {amount})")
    else:
        await page.evaluate("window.scrollBy(0, window.innerHeight)")
    return "Scrolled down"

@mcp.tool()
async def scroll_up(amount: int = None) -> str:
    """Scroll up the page."""
    page = await browser_context.get_current_page()
    if amount:
        await page.evaluate(f"window.scrollBy(0, -{amount})")
    else:
        await page.evaluate("window.scrollBy(0, -window.innerHeight)")
    return "Scrolled up"

# IPL-specific functions
@mcp.tool()
async def get_current_matches() -> Dict[str, Any]:
    """Get list of all current IPL matches."""
    try:
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
        if not browser_context:
            await initialize_browser(headless=False, task="Get Points Table")
        
        await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/points-table")
        await wait(2)
        
        table_content = await inspect_page()
        return {"status": "success", "points_table": table_content}
    except Exception as e:
        logger.error(f"Error getting points table: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_stats() -> Dict[str, Any]:
    """Get IPL statistics (batting and bowling)."""
    try:
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
        except:
            logger.warning("Could not load bowling stats")
        
        return {"status": "success", "stats": stats_content}
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return {"status": "error", "message": str(e)}

@mcp.tool()
async def get_team_schedule(team_name: str) -> Dict[str, Any]:
    """Get schedule for a specific IPL team."""
    try:
        if not browser_context:
            await initialize_browser(headless=False, task="Get Team Schedule")
        
        await go_to_url("https://www.cricbuzz.com/cricket-series/9237/indian-premier-league-2025/matches")
        await wait(2)
        
        schedule_content = await inspect_page()
        return {
            "status": "success",
            "team": team_name,
            "schedule": schedule_content
        }
    except Exception as e:
        logger.error(f"Error getting team schedule: {str(e)}")
        return {"status": "error", "message": str(e)}

def main():
    """Run the MCP server for IPL information."""
    logger.info("Starting MCP server for IPL information")
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main() 
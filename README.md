# IPL Data Automation with MCP

A Python-based automation tool that uses MCP (Model Control Protocol) and browser automation to fetch and cache IPL cricket data. The system minimizes browser calls by implementing an JSON-based caching system.

## Quick Setup

1. Install dependencies:
```bash
pip install mcp-server browser-use playwright python-dotenv
playwright install
```

2. Create cache directory:
```bash
mkdir -p cache
```

3. Start the server:
```bash
python main.py
```

## Cache System

The system uses `ipl_cache.json` to store data with the following structure:

```json
{
    "schedule": {
        "lastUpdated": "2024-03-21T10:00:00",
        "matches": []
    },
    "pointsTable": {
        "lastUpdated": "2024-03-21T10:00:00",
        "teams": []
    },
    "lastChecked": {
        "schedule": "2024-03-21T10:00:00",
        "pointsTable": "2024-03-21T10:00:00"
    }
}
```

### Cache Update Frequency
- Points Table: Daily updates
- Schedule: Weekly updates
- Live Matches: Every 5 minutes during matches

### Browser Optimization
The system minimizes browser calls by:
1. Checking cache before initializing browser
2. Using `lastChecked` timestamps to prevent frequent checks
3. Reusing browser sessions when multiple data types are requested
4. Implementing graceful fallback to cached data

## Available Functions

```python
# Get IPL Schedule (checks cache first, updates weekly)
await get_ipl_schedule()

# Get Points Table (checks cache first, updates daily)
await get_points_table()

# Get Team Schedule (uses cached schedule data)
await get_team_schedule("Mumbai Indians")

# Get Live Match Details (real-time updates)
await get_match_details("match_id")
```

## Testing Setup

Run the test script to verify everything is working:
```bash
python test_setup.py
```

This will test:
- Browser initialization
- Cache system
- Data fetching
- Error handling

## Troubleshooting

1. **Browser Issues**:
   ```bash
   # Reinstall browsers
   playwright install --force
   ```

2. **Cache Issues**:
   ```bash
   # Reset cache
   rm cache/ipl_cache.json
   python main.py
   ```

3. **Common Errors**:
   - "Browser not initialized": Check if browser-use is properly installed
   - "Cache file not found": Ensure cache directory exists
   - "Data fetch failed": Check internet connection and Cricbuzz accessibility

## Environment Variables

Create `.env` file:
```env
DEBUG=True
HEADLESS=True
CACHE_DIR=./cache
BROWSER_TIMEOUT=30000
```

## Logs

Check `logs/test_setup.log` for detailed debugging information.
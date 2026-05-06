const fs = require('fs');
const Papa = require('papaparse');

const csvData = fs.readFileSync('005930_KS_OHLCV.csv', 'utf8');

Papa.parse(csvData, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
    complete: function(results) {
        let parsed = [];
        for(let row of results.data) {
            let dateStr = row.Date || row.Datetime || row.time;
            if(!dateStr) continue;
            
            try {
                let timeStr = new Date(dateStr).toISOString().split('T')[0];
                parsed.push({
                    time: timeStr,
                    open: row.Open,
                    high: row.High,
                    low: row.Low,
                    close: row.Close
                });
            } catch(err) { console.error("Date error:", err); continue; }
        }
        
        parsed.sort((a,b) => new Date(a.time) - new Date(b.time));
        
        console.log("Parsed rows:", parsed.length);
        if(parsed.length > 0) {
            console.log("First row:", parsed[0]);
            console.log("Last row:", parsed[parsed.length-1]);
        }
        
        // Check for duplicates
        let times = new Set();
        let hasDupes = false;
        for(let p of parsed) {
            if(times.has(p.time)) {
                console.log("Duplicate time found:", p.time);
                hasDupes = true;
                break;
            }
            times.add(p.time);
        }
        console.log("Has duplicates:", hasDupes);
    }
});

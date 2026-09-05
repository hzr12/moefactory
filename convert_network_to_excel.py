import json
import pandas as pd
from pathlib import Path

def main():
    # Read JSON
    json_path = Path("datasets/network.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    lines = data["lines"]
    
    # Create Excel writer
    output_path = Path("outputs/network.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for line_name, stations in lines.items():
            # Create DataFrame for this line
            df = pd.DataFrame(stations, columns=["车站", "里程"])
            
            # Write to sheet (truncate sheet name to 31 chars for Excel limit)
            sheet_name = line_name[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Created {output_path} with {len(lines)} sheets")

if __name__ == "__main__":
    main()
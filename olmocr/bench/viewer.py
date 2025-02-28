#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict

from olmocr.data.renderpdf import render_pdf_to_base64png


def parse_rules_file(file_path):
    """Parse the rules file and organize rules by PDF."""
    pdf_rules = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                rule = json.loads(line)
                if "pdf" in rule:
                    pdf_rules[rule["pdf"]].append(rule)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")

    return pdf_rules


def get_rule_html(rule):
    """Generate HTML representation for a rule."""
    rule_type = rule.get("type", "unknown")

    if rule_type == "present":
        return f"""
        <tr class="rule-row present-rule">
            <td><span class="rule-type present">PRESENT</span></td>
            <td>"{rule.get('text', '')}"</td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    elif rule_type == "absent":
        return f"""
        <tr class="rule-row absent-rule">
            <td><span class="rule-type absent">ABSENT</span></td>
            <td>"{rule.get('text', '')}"</td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    elif rule_type == "order":
        return f"""
        <tr class="rule-row order-rule">
            <td><span class="rule-type order">ORDER</span></td>
            <td>
                <p><strong>Before:</strong> "{rule.get('before', '')}"</p>
                <p><strong>After:</strong> "{rule.get('after', '')}"</p>
            </td>
            <td>Threshold: {rule.get('threshold', 'N/A')}</td>
        </tr>
        """
    else:
        return f"""
        <tr class="rule-row unknown-rule">
            <td><span class="rule-type unknown">UNKNOWN</span></td>
            <td>Unknown rule type: {rule_type}</td>
            <td></td>
        </tr>
        """


def generate_html(pdf_rules, rules_file_path):
    """Generate the HTML page with PDF renderings and rules."""
    # Limit to 10 unique PDFs
    pdf_names = list(pdf_rules.keys())[:10]

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Rules Visualizer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            
            .container {
                max-width: 1600px;
                margin: 0 auto;
            }
            
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            
            .pdf-container {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
                overflow: hidden;
            }
            
            .pdf-header {
                background-color: #4a6fa5;
                color: white;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
            
            .pdf-content {
                display: flex;
                flex-direction: row;
                padding: 20px;
            }
            
            @media (max-width: 1200px) {
                .pdf-content {
                    flex-direction: column;
                }
            }
            
            .pdf-image {
                flex: 0 0 50%;
                max-width: 800px;
                text-align: center;
                padding-right: 20px;
            }
            
            .pdf-image img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            
            .rules-container {
                flex: 1;
                overflow: auto;
            }
            
            .rules-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .rules-table th {
                background-color: #4a6fa5;
                color: white;
                padding: 10px;
                text-align: left;
            }
            
            .rules-table td {
                padding: 10px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }
            
            .rule-type {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
            }
            
            .present {
                background-color: #28a745;
            }
            
            .absent {
                background-color: #dc3545;
            }
            
            .order {
                background-color: #fd7e14;
            }
            
            .unknown {
                background-color: #6c757d;
            }
            
            .rule-row:hover {
                background-color: #f8f9fa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF Rules Visualizer</h1>
    """

    for pdf_name in pdf_names:
        rules = pdf_rules[pdf_name]

        # Render the PDF (first page only) from the /pdfs folder
        try:
            pdf_path = os.path.join(os.path.dirname(rules_file_path), "pdfs", pdf_name)
            base64_img = render_pdf_to_base64png(pdf_path, 0)
            img_html = f'<img src="data:image/png;base64,{base64_img}" alt="{pdf_name}">'
        except Exception as e:
            img_html = f'<div class="error">Error rendering PDF: {str(e)}</div>'

        html += f"""
        <div class="pdf-container">
            <div class="pdf-header">{pdf_name}</div>
            <div class="pdf-content">
                <div class="pdf-image">
                    {img_html}
                </div>
                <div class="rules-container">
                    <table class="rules-table">
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th>Content</th>
                                <th>Parameters</th>
                            </tr>
                        </thead>
                        <tbody>
        """

        for rule in rules:
            html += get_rule_html(rule)

        html += """
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate an HTML visualization of PDF rules.")
    parser.add_argument("rules_file", help="Path to the rules file (JSON lines format)")
    parser.add_argument("-o", "--output", help="Output HTML file path", default="pdf_rules_visualization.html")

    args = parser.parse_args()

    if not os.path.exists(args.rules_file):
        print(f"Error: Rules file not found: {args.rules_file}")
        sys.exit(1)

    pdf_rules = parse_rules_file(args.rules_file)
    html = generate_html(pdf_rules, args.rules_file)

    with open(args.output, "w") as f:
        f.write(html)

    print(f"HTML visualization created: {args.output}")


if __name__ == "__main__":
    main()

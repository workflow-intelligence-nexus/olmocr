#!/usr/bin/env python3
"""
Render LaTeX equations to PNG images using Playwright and KaTeX

Requirements:
    pip install playwright
    python -m playwright install chromium
    
    Place katex.min.css and katex.min.js in the same directory as this script
"""

import os
from playwright.sync_api import sync_playwright

def render_equation_to_png(
    equation, 
    output_path="equation.png", 
    bg_color="white",
    text_color="black",
    font_size=24,
):
    """
    Render a LaTeX equation to a PNG file using Playwright and KaTeX.
    
    Args:
        equation (str): LaTeX equation to render
        output_path (str): Path to save the PNG file
        bg_color (str): Background color
        text_color (str): Text color
        font_size (int): Font size in pixels
    """
    # We need to escape backslashes for JavaScript string
    escaped_equation = equation.replace("\\", "\\\\")
    
    # Get the directory of the script to reference local files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    katex_css_path = os.path.join(script_dir, "katex.min.css")
    katex_js_path = os.path.join(script_dir, "katex.min.js")
    
    # Check if the files exist
    if not os.path.exists(katex_css_path) or not os.path.exists(katex_js_path):
        raise FileNotFoundError(f"KaTeX files not found. Please ensure katex.min.css and katex.min.js are in {script_dir}")
    
    with sync_playwright() as p:
        # Launch a headless browser
        browser = p.chromium.launch()
        
        # Create a new page with a reasonable viewport size
        page = browser.new_page(viewport={"width": 800, "height": 400})
        
        # Basic HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: {bg_color};
                    color: {text_color};
                }}
                #equation-container {{
                    padding: 0px;
                    font-size: {font_size}px;
                }}
            </style>
        </head>
        <body>
            <div id="equation-container"></div>
        </body>
        </html>
        """
        
        # Set the page content
        page.set_content(html)
        
        # Add KaTeX CSS and JS files
        page.add_style_tag(path=katex_css_path)
        page.add_script_tag(path=katex_js_path)
        
        page.wait_for_load_state("networkidle")
        
        # Check if KaTeX is properly loaded
        katex_loaded = page.evaluate("typeof katex !== 'undefined'")
        if not katex_loaded:
            raise RuntimeError("KaTeX library failed to load. Check your katex.min.js file.")
        
        # Render the equation
        page.evaluate(f"""
        () => {{
            katex.render("{escaped_equation}", document.getElementById("equation-container"), {{
                displayMode: true,
                throwOnError: false
            }});
        }}
        """)
        
        # Wait for the equation to be rendered
        page.wait_for_selector(".katex", state="attached")
        
        # Get the container element and take a screenshot
        container = page.query_selector("#equation-container")
        
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Take the screenshot
        container.screenshot(path=output_path)
        
        # Close the browser
        browser.close()
        
        return output_path

def main():
    # Example equation: Einstein's famous equation
    simple_equation = "E = mc^2"
    
    # More complex equation: Quadratic formula
    complex_equation = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
    
    # Maxwell's equations in differential form
    maxwell_equation = "\\begin{aligned} \\nabla \\cdot \\vec{E} &= \\frac{\\rho}{\\epsilon_0} \\\\ \\nabla \\cdot \\vec{B} &= 0 \\\\ \\nabla \\times \\vec{E} &= -\\frac{\\partial\\vec{B}}{\\partial t} \\\\ \\nabla \\times \\vec{B} &= \\mu_0 \\vec{J} + \\mu_0\\epsilon_0\\frac{\\partial\\vec{E}}{\\partial t} \\end{aligned}"
    
    # Render the equations
    render_equation_to_png(simple_equation, "einstein_equation.png")
    render_equation_to_png(complex_equation, "quadratic_formula.png")
    render_equation_to_png(
        maxwell_equation, 
        "maxwell_equations.png")
    
    print("All equations rendered successfully!")

if __name__ == "__main__":
    main()
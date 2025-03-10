#!/usr/bin/env python3
"""
Render LaTeX equations to Pillow images using Playwright and KaTeX
with SHA1-based caching mechanism.

Requirements:
    pip install playwright pillow
    python -m playwright install chromium
    
    Place katex.min.css and katex.min.js in the same directory as this script
"""

import os
import hashlib
import pathlib

from PIL import Image
from playwright.sync_api import sync_playwright, Error as PlaywrightError

def get_equation_hash(equation, bg_color="white", text_color="black", font_size=24):
    """
    Calculate SHA1 hash of the equation string and rendering parameters.
    
    Args:
        equation (str): LaTeX equation to hash
        bg_color (str): Background color
        text_color (str): Text color
        font_size (int): Font size in pixels
        
    Returns:
        str: SHA1 hash of the equation and parameters
    """
    # Combine all parameters that affect the output into a single string
    params_str = f"{equation}|{bg_color}|{text_color}|{font_size}"
    return hashlib.sha1(params_str.encode('utf-8')).hexdigest()

def get_cache_dir():
    """
    Get the cache directory for equations, creating it if it doesn't exist.
    
    Returns:
        pathlib.Path: Path to the cache directory
    """
    cache_dir = pathlib.Path.home() / '.cache' / 'olmocr' / 'bench' / 'equations'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

def render_equation(
    equation, 
    bg_color="white",
    text_color="black",
    font_size=24,
    use_cache=True,
    debug_dom=False,
):
    """
    Render a LaTeX equation to a Pillow Image using Playwright and KaTeX.
    Uses caching based on SHA1 hash of the equation.
    
    Args:
        equation (str): LaTeX equation to render
        bg_color (str): Background color
        text_color (str): Text color
        font_size (int): Font size in pixels
        use_cache (bool): Whether to use caching
        debug_dom (bool): Whether to print the KaTeX DOM structure
        
    Returns:
        PIL.Image.Image: Pillow image of the rendered equation
    """
    # Calculate the equation's hash for caching, including all rendering parameters
    eq_hash = get_equation_hash(equation, bg_color, text_color, font_size)
    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{eq_hash}.png"
    cache_error_file = cache_dir / f"{eq_hash}_error"
    
    # Check if the equation is already cached
    if use_cache:
        if cache_error_file.exists():
            return None
            
        if cache_file.exists():
            return Image.open(cache_file)

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
        
        # Render the equation and check for errors
        try:
            has_error = page.evaluate(f"""
            () => {{
                try {{
                    katex.render("{escaped_equation}", document.getElementById("equation-container"), {{
                        displayMode: true,
                        throwOnError: true
                    }});
                    return false; // No error
                }} catch (error) {{
                    console.error("KaTeX error:", error.message);
                    return true; // Error occurred
                }}
            }}
            """)
        except PlaywrightError as ex:
            has_error = True

        if has_error:
            print(f"Error rendering equation: '{equation}'")
            cache_error_file.touch()
            browser.close()
            return None
        
        # Wait for the equation to be rendered
        page.wait_for_selector(".katex", state="attached")
        
        # Extract and print KaTeX DOM HTML if debug_dom is enabled
        if debug_dom:
            # Get the HTML structure of the rendered equation
            katex_dom_html = page.evaluate("""
            () => {
                const container = document.getElementById("equation-container");
                return container.innerHTML;
            }
            """)
            
            # Print the KaTeX DOM HTML
            print("\n===== KaTeX DOM HTML =====")
            print(katex_dom_html)
        
        # Get the container element and take a screenshot
        container = page.query_selector("#equation-container")
        
        # Take the screenshot
        container.screenshot(path=str(cache_file))
        
        # Close the browser
        browser.close()
        
        # Return the image as a Pillow Image
        return Image.open(cache_file)

def main():
    # Example equation: Einstein's famous equation
    simple_equation = "E = mc^2"
    
    # More complex equation: Quadratic formula
    complex_equation = "x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}"
    
    # Maxwell's equations in differential form
    maxwell_equation = "\\begin{aligned} \\nabla \\cdot \\vec{E} &= \\frac{\\rho}{\\epsilon_0} \\\\ \\nabla \\cdot \\vec{B} &= 0 \\\\ \\nabla \\times \\vec{E} &= -\\frac{\\partial\\vec{B}}{\\partial t} \\\\ \\nabla \\times \\vec{B} &= \\mu_0 \\vec{J} + \\mu_0\\epsilon_0\\frac{\\partial\\vec{E}}{\\partial t} \\end{aligned}"
    
    # Render the equations
    # Default parameters
    bg_color = "white"
    text_color = "black"
    font_size = 24
    
    print("========== Rendering Einstein's Equation ==========")
    image1 = render_equation(simple_equation, bg_color, text_color, font_size)
    image1.save("einstein_equation.png")
    print(f"Einstein's equation hash: {get_equation_hash(simple_equation, bg_color, text_color, font_size)}")
    
    print("\n========== Rendering Quadratic Formula ==========")
    image2 = render_equation(complex_equation, bg_color, text_color, font_size)
    image2.save("quadratic_formula.png")
    print(f"Quadratic formula hash: {get_equation_hash(complex_equation, bg_color, text_color, font_size)}")
    
    # Different styling for Maxwell's equations
    maxwell_bg = "black"
    maxwell_text = "white"
    maxwell_size = 20
    
    print("\n========== Rendering Maxwell's Equations ==========")
    image3 = render_equation(maxwell_equation, maxwell_bg, maxwell_text, maxwell_size)
    image3.save("maxwell_equations.png")
    print(f"Maxwell's equations hash: {get_equation_hash(maxwell_equation, maxwell_bg, maxwell_text, maxwell_size)}")
    
    # Example of retrieving from cache with same parameters
    print("\n========== Retrieving Einstein's Equation from Cache ==========")
    # Set debug_dom to False for cached version to avoid duplicate debug output
    image_from_cache = render_equation(simple_equation, bg_color, text_color, font_size, debug_dom=False)
    print("Retrieved Einstein's equation from cache.")
    
    # Example of different styling for the same equation (will render and cache separately)
    alt_bg = "lightblue"
    alt_text = "darkblue"
    alt_size = 30
    
    print("\n========== Rendering Einstein's Equation with Alternate Style ==========")
    image_alt_style = render_equation(simple_equation, alt_bg, alt_text, alt_size)
    image_alt_style.save("einstein_equation_alt_style.png")
    print(f"Einstein's equation with alternate style hash: {get_equation_hash(simple_equation, alt_bg, alt_text, alt_size)}")

    print("\n========== Rendering Invalid Equation ==========")
    invalid = render_equation("$150. \\quad s(t) = 2t^3 - 3t^2 - 12t + 8")
    
    print("\nAll equations rendered successfully!")

if __name__ == "__main__":
    main()
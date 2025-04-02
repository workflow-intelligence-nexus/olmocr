import unittest

from .mine_html_templates import generate_tests_from_html

class TestMineTests(unittest.TestCase):
    def test_absent_nested(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>New Paradigm for Nuclear Safety</title>
</head>
<body>
    <header>
        <div class="logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
    </header>
    
    <main>
        <h1>New Paradigm for Nuclear Safety</h1>
        
        <div class="attribution">
            <p>Thursday, April 25, 2013</p>
            <p>Japan Nuclear Safety Institute</p>
            <p>Shojiro Matsuura, Chairman</p>
        </div>
    </main>
    
    <footer>
        <div class="footer-line"></div>
        <div class="footer-content">
            <div class="footer-text">
                <p class="tagline">In Pursuit of Improved Nuclear Safety</p>
                <p class="copyright">Copyright © 2012 by Japan Nuclear Safety Institute. All Rights Reserved.</p>
            </div>
            <div class="footer-logo" aria-label="Japan Nuclear Safety Institute Logo"></div>
        </div>
    </footer>
</body>
"""
        tests = generate_tests_from_html(html_content, "0", 1)

        self.assertEqual(len([test for test in tests if test["type"]=="absent"]), 2)
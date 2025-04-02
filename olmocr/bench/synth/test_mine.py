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

    def test_text_basic(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bone Morphology Description</title>
</head>
<body>
    <main>
        <p>The posterior end exhibits a curved process to articulate with the angular. Aside from the process, the rest of the posterior end has slight curvatures for articulation, but is mostly uniform. Ventral border of the bone is mostly straight, with slight curvature (FIG. 20).</p>
        
        <p><span class="section-heading">Lateral</span>- A spine runs from the anterior-most tip, reduces in height ~3/4 way down toward posterior, and terminates at the center of the posterior notch. A fossa is present on the dorsal side of the spine. The posterior end exhibits more relief than in medial view, with the medial side of the posterior process extending past the posterior notch.</p>
        
        <p><span class="section-heading">Ontogeny</span>- Anterior tip is sharply pointed in AR12 and AR1 with AR2 being rounded, though this could be due to breakage. Anterior dorsal margin is straight and flat in AR12; AR2 shows little curvature and AR1 shows the most curvature; curving outward dorsally. Dorsal incisure is anteroposteriorly oriented in AR12, in AR2 there is some ventral curvature, and in AR1 there is a posteroventral curvature. Both AR1 and AR3 are curved on the ventral margin while AR12 is mostly straight. Posterior end of AR1 exhibits four undulations, ventral process is not yet extended. A fossa is present dorsal to the ventral process, not seen on AR12 or AR2. In medial view the lateral ridge is visible posteriorly in AR1 and AR2l the ridge does not fully extend anteriorly. In lateral view of the posterior the ventral process is present on AR2, but not fully extended posteriorly. Tip of the anterodorsal process is sharply pointed in AR1 and AR2, rounded in AR12. A short ridge is present on the dorsal edge of the dorsal process of AR1. The short ridge on the posterodorsal process of AR2 is slightly more ventral than in AR1. On AR12 the ridge is long and positioned most ventral. The lateral ridge is closest to the ventral margin in AR1. In AR2 the ridge is positioned more dorsally and in AR12 the ridge terminates and the anterior tip. The section of bone ventral to the lateral ridge appears to thin with age. The posterior notch on AR12 is curved anteriorly and the medial side of the notch extends posteriorly</p>
    </main>
    
    <footer>
        <p>46</p>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)
        self.assertGreater(len(tests), 5)

    def test_big_headers(self):
        html_content = """

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FORANE 427A Comparative Data</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 724px;
            margin: 0 auto;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        table, th, td {
            border: 1px solid black;
        }
        
        th, td {
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        .logo {
            width: 150px;
            height: 80px;
            background-color: #eee;
            border: 1px solid #000;
            margin-top: 20px;
        }
        
        footer {
            margin-top: 40px;
            font-size: 0.8em;
        }
        
        .company-info {
            margin-top: 20px;
            display: flex;
            justify-content: space-between;
        }
        
        .contact {
            text-align: right;
        }
    </style>
</head>
<body>
    <header>
        <h1>Comparative data</h1>
    </header>
    
    <main>
        <table>
            <thead>
                <tr>
                    <th>Parameters</th>
                    <th>R-22</th>
                    <th>FORANE<sup>®</sup> 427A</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Evaporating temperature</td>
                    <td>2.7 °C</td>
                    <td>1.1 °C</td>
                </tr>
                <tr>
                    <td>Condensing temperature</td>
                    <td>40.3 °C</td>
                    <td>44.0 °C</td>
                </tr>
                <tr>
                    <td>Suction temperature</td>
                    <td>7.1 °C</td>
                    <td>9.2 °C</td>
                </tr>
                <tr>
                    <td>Suction pressure</td>
                    <td>5.4 bar</td>
                    <td>5.0 bar</td>
                </tr>
                <tr>
                    <td>Discharge temperature</td>
                    <td>69.5 °C</td>
                    <td>71.1 °C</td>
                </tr>
                <tr>
                    <td>Discharge pressure</td>
                    <td>15.5 bar</td>
                    <td>17.1 bar</td>
                </tr>
                <tr>
                    <td>Cooling power</td>
                    <td>431 KW</td>
                    <td>376 KW</td>
                </tr>
                <tr>
                    <td>Power consumption</td>
                    <td>122 kW</td>
                    <td>124 kW</td>
                </tr>
                <tr>
                    <td>Residual mineral oil</td>
                    <td>-</td>
                    <td>11%</td>
                </tr>
            </tbody>
        </table>
        
        <p>During this field test, very satisfactory running conditions were reached immediately. The temperature set points were easily achieved with similar energy consumption as compared to R-22 despite a high level of residual mineral oil in the circuit. The performance of the installation continues to satisfy the customer's requirements after more than one year of service.</p>
        
        <p>FORANE<sup>®</sup> 427A consequently fully satisfies the requirements of the European regulations while enabling existing equipment to continue to perform well without the need for any long and costly plant modifications.</p>
        
        <p>The versatility of FORANE<sup>®</sup> 427A is also appreciated as it can be used to retrofit low temperature refrigeration equipment as well as air-conditioning installations, resulting in only one retrofit refrigerant for all R-22 units.</p>
        
        <p>Combining environmental friendliness, high performance and simplicity is today a reality with FORANE<sup>®</sup> 427A !</p>
    </main>
    
    <footer>
        <p>The information contained in this document is based on trials carried out by our Research Centres and data selected from the literature, but shall in no event be held to constitute or imply any warranty, undertaking, express or implied commitment from our part. Our formal specifications define the limit of our commitment. No liability whatsoever can be accepted by Arkema with regard to the handling, processing or use of the product or products concerned which must in all cases be employed in accordance with all relevant laws and/or regulations in force in the country or countries concerned.</p>
        
        <div class="company-info">
            <div class="address">
                <div class="logo"></div>
                <p>ARKEMA<br>
                420 rue d'Estienne d'Orves<br>
                92700 Colombes - France<br>
                www.arkema.com</p>
            </div>
            
            <div class="contact">
                <p>www.forane.com / info.forane@arkema.com</p>
            </div>
        </div>
    </footer>
</body>
</html>"""

        tests = generate_tests_from_html(html_content, "0", 1)

        for test in tests:
            print(test)

        self.assertFalse(any(test for test in tests if test["type"] == "absent" and "Comparative data" in test["text"]))
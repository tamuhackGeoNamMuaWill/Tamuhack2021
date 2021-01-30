#!/usr/bin/python
# Import modules for CGI handling
import cgi, cgitb
# Create instance of FieldStorage
form = cgi.FieldStorage()
# Get data from fields
first_name = form.getvalue('first_name')
last_name = form.getvalue('last_name')
text = '''
<html>
    <body>
        <h1>Heading (first_name, last_name) </h1> 
    </body>
</html>
'''
file = open("pythonReturns.html","w")
file.write(text)
file.close()

#!/usr/bin/python
# Import modules for CGI handling
import cgi, cgitb, bs4
# Create instance of FieldStorage
form = cgi.FieldStorage()
# Get data from fields
first_name = form.getvalue('first_name')
last_name = form.getvalue('last_name')


# load the file
with open("pythonReturns.html") as inf:
    txt = inf.read()
    soup = bs4.BeautifulSoup(txt)

# create new link
new_link = soup.new_tag("link", rel="icon", type="image/png", href="img/tor.png")
# insert it into the document
soup.head.append(new_link)

# save the file again
with open("existing_file.html", "w") as outf:
    outf.write(str(soup))

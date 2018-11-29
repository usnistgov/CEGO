"""
** For testing only **

First, in this folder:
git clone https://github.com/openjournals/whedon

Then run this file in python.  

"""
cmd = """pandoc  \
-V paper_title="TITLE GOES HERE" \
-V citation_author="Ian H. Bell" \
-V archive_doi="http://dx.doi.org/00.00000/zenodo.0000000" \
-V formatted_doi="00.00000/joss.00000" \
-V paper_url="http://joss.theoj.org/papers/" \
-V review_issue_url="http://joss.theoj.org/papers/" \
-V issue="0" \
-V volume="00" \
-V year="2018" \
-V submitted="00 January 0000" \
-V published="00 January 0000" \
-V page="00" \
-V graphics="true" \
-V logo_path="whedon/resources/joss-logo.png" \
-V geometry:margin=1in \
-o paper.pdf \
--pdf-engine=xelatex \
--filter pandoc-citeproc paper.md \
--from markdown+autolink_bare_uris \
--template "whedon/resources/latex.template"
""".replace('\\','')

import subprocess, sys
subprocess.check_call(cmd, shell=True, stdout=sys.stdout,stderr=sys.stderr)
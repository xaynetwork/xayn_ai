#!/usr/bin/env python3
from http import server

class MyHTTPRequestHandler(server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")

        server.SimpleHTTPRequestHandler.end_headers(self)

def main(data_dir):
    import os
    os.chdir(data_dir)
    print("Now hosting web example at http://localhost:8000")
    print("DO NOT USE AN IP ADDRESS TO OPEN THE WEB SITE, IT WILL NOT WORK")
    httpd = server.HTTPServer(('localhost', 8000), MyHTTPRequestHandler)
    httpd.serve_forever()

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2 and sys.argv[1] not in ["-h", "--help"]:
        main(data_dir=sys.argv[1])
    else:
        print(f"Usage {sys.argv[0]} <data_dir>", file=sys.stderr)

import os
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import requests
import re
from urllib.request import urlretrieve
from pathlib import Path
import PyPDF2
from tika import parser
import tensorflow as tf
import numpy as np
import random
import sys
import io
import slate3k as slate
import time
import warnings

Curr = {}
ll=0
class DataProcessing:

    def __init__(self, TopLevel, desFolder, NumPages, keyPhrase):
        self.TopLevel = TopLevel
        self.desFolder = desFolder
        self.NumPages = NumPages
        self.keyPhrase = keyPhrase
        self.SafeMake(self.desFolder)

    def Scrawl(self):
        for p in range(self.NumPages):
        
            http = httplib2.Http()

            if p == 0:
                status, response = http.request(self.TopLevel)
            else:
                status, response = http.request(self.TopLevel + '?_paged=' + str(p))
            
            for ii,link in enumerate(BeautifulSoup(response, 'html.parser',
                                         parseOnlyThese=SoupStrainer('a'))):
                if link.has_attr('href'):
                    if self.keyPhrase in link['href'] and not link['href'] in Curr:
                        Curr.update({link['href']: 1})
                        print(link['href'])
                        name = link['href'].split('/')[-2].replace('transcript-mbmbam-','').replace("-","_")
                        r = requests.get(link['href'])
                        soup = BeautifulSoup(r.text, "html.parser")
                        for y in [list(map(int, re.findall(r'\d+', x))) for x in os.listdir(self.desFolder)]:
                            if list(map(int, re.findall(r'\d+', name)))[0] in y:
                                return
                        desFile = os.path.join(self.desFolder, str(name) + '.pdf')
                        f = Path(desFile)
                        for i in soup.find_all('a', href=True):
                            if ".pdf" in i.attrs['href']:
                                ri = requests.get(i['href'])
                                f.write_bytes(ri.content)

        return
        
    def SafeMake(self, folder):
        try:
            return os.makedirs(folder)
        except FileExistsError:
            return Path(folder)
    
    
    def TextIt(self):
        Brothers = os.path.join(os.getcwd(), "Brothers")
        TextFiles = os.path.join(os.getcwd(), "TextFiles")
        self.SafeMake(TextFiles)
        for file in os.listdir(Brothers):
            if os.path.join(TextFiles, file.replace("pdf","txt")) in os.listdir(TextFiles):
                pass
            oFile = open(os.path.join(TextFiles, file.replace("pdf","txt")), "w+", encoding="utf-8")
            #raw = parser.from_file(os.path.join(Brothers,file))
            #raw['content'].replace('\n','')
            pdfFileObj = open(os.path.join(Brothers,file),"rb")
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
            for page in range(pdfReader.numPages):
                pageObj = pdfReader.getPage(page)
                body = pageObj.extractText()
                SeparatedBody = [s.replace("\n","") + "\n" for s in body.split(".")]
                for line in SeparatedBody:
                    if len(line) > 5:
                        if "..." in line:
                            line = line.replace("...","BIGBONGFGH")
                        oFile.write(line.replace(".",".\n").replace("BIGBONGFGH", "..."))
            oFile.close()
        return
    
    def TextIt2(self):
        Brothers = os.path.join(os.getcwd(), "Brothers")
        TextFiles = os.path.join(os.getcwd(), "TextFiles")
        self.SafeMake(TextFiles)
        for file in os.listdir(Brothers):
            oFile = open(os.path.join(TextFiles, file.replace("pdf","txt")), "w+", encoding="utf-8")
            iFile = open(os.path.join(Brothers, file), "rb")
            doc = slate.PDF(iFile)
            for page in doc:
                page = re.sub(r'\n+ ', '\n', page)
                page = re.sub(r'\n+', '\n', page)
                page = re.sub(r'[^\x00-\x7F]+','', page)
                #page = re.sub(r'[^0-9a-zA-Z]+','', page)
                if len(page)> 6: oFile.write(page[:-1])
            iFile.close()
            oFile.close()
        return
    
    def CombineIt(self):
        TextFiles = SafeMake(os.path.join(os.getcwd(), "TextFiles"))
        oFile = open(os.path.join(TextFiles,"Beeg.txt"), "w+")
        for file in os.listdir(TextFiles)[0:10]:
            if not file == oFile:
                iFile = open(os.path.join(TextFiles,file),"r+")
                lines = iFile.readlines()
                for line in lines:
                    oFile.write(line)
                    #oFile.write(line.replace("\n","").replace(".","").replace(",","").replace("!","").replace(":",""))
                iFile.close()
        oFile.close()
        return
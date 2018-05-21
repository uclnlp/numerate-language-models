from __future__ import print_function
from lxml import etree
import re
from pprint import pprint


NS = {'ns': "http://dlmf.nist.gov/LaTeXML"}


def get_xml_string(node):
    return etree.tostring(node, method='xml', encoding='unicode', pretty_print=True)


def get_only_text(node):
    # replace bibrefs
    for bibref in node.xpath('//ns:bibref[@bibrefs]', namespaces=NS):
        refs = ['_cite_' for ref in bibref.attrib.get('bibrefs', '').strip().split(',')]
        new_node = etree.XML("<div> " + ', '.join(refs) + " </div>")
        new_node.tail = bibref.tail
        parent = bibref.getparent()
        parent.replace(bibref, new_node)
    # replace label refs
    for labelref in node.xpath('//ns:ref[@labelref]', namespaces=NS):
        new_node = etree.XML("<div> " + '_labelref_' + " </div>")
        new_node.tail = labelref.tail
        parent = labelref.getparent()
        parent.replace(labelref, new_node)
    # replace math
    for texmath in node.xpath('//ns:math[@tex]', namespaces=NS):
        tex = texmath.attrib.get('tex', '').strip()
        tex = tex.replace('_', ' _ ')
        new_node = etree.XML("<div> _s_math_ " + tex + " _e_math_ </div>")
        new_node.tail = texmath.tail
        parent = texmath.getparent()
        parent.replace(texmath, new_node)
    txt = etree.tostring(node, method='text', encoding='unicode')
    txt = txt.replace('\n', ' ')
    txt = re.sub(r'\s+', ' ', txt)
    txt = re.sub(r'(\r?\n)+', '\n', txt)
    txt = txt.strip()
    #print(txt)
    return txt

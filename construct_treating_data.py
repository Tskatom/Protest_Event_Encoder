#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os

def main():
    infolder = "./data/single_label"
    outfolder = "./data/treating_single_label"
    prefix = "spanish_protest"

    treating_sens = {"011": "Los hospitales de Suchitoto y Sonsonate se encuentran en reducción de labores . ",
            "012": "PROTESTA | Exigen viviendas dignas al gobernador García Carneiro Vargas colapsada por protesta de damnificados de un refugio Trancaron el paso hacia la capital en la autopista . ", 
            "013": "Vecinos de Los Teques protestan por fallas en el suministro de gas La protesta mantiene congestionada . ",
            "014": "Diversos comercios de la ciudad manifestaron afectaciones por el incremento en el precio por caja . ",
            "015": "Ciudades Marcha para evitar que se lleven el monumento a Colón EL TRASLADO FUE ACORDADO POR LOS . ",
            "016": "Cientos de fieles católicos , la mayoría de ellos habitantes del barrio de Flores . "
            }
   
    groups = ["train", "valid", "test"]
    for g in groups:
        infile = os.path.join(infolder, "%s_%s.txt.tok" % (prefix, g))
        outfile = os.path.join(outfolder, "%s_%s.txt.tok" % (prefix, g))
        
        in_y = os.path.join(infolder, "%s_%s.type_cat" % (prefix, g))
        out_y = os.path.join(outfolder, "%s_%s.type_cat" % (prefix, g))
        
        print infolder, "%s_%s.type_cat" % (prefix, g)

        with open(infile) as ifi, open(outfile, 'w') as otf, open(in_y) as iny, open(out_y, 'w') as outy:
            docs = [line for line in ifi]
            tags = [tag.strip() for tag in iny]
            for doc, tag in zip(docs, tags):
                add_sen = treating_sens[tag]
                doc = add_sen + doc
                otf.write(doc)

                outy.write("%s\n" % tag)


if __name__ == "__main__":
    main()

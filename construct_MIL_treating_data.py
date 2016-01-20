#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"

import sys
import os
import re
import random

def main():
    infolder = "./data/single_label"
    outfolder = "./data/treating_MIL_single_label"
    prefix = "spanish_protest"

    treating_sens = {"011": "Los hospitales de Suchitoto y Sonsonate se encuentran en reducción de labores . ",
            "012": "Exigen viviendas dignas al gobernador García Carneiro Vargas colapsada por protesta de damnificados de un refugio Trancaron el paso hacia la capital en la autopista . ", 
            "013": "Vecinos de Los Teques protestan por fallas en el suministro de gas La protesta mantiene congestionada . ",
            "014": "Diversos comercios de la ciudad manifestaron afectaciones por el incremento en el precio por caja . ",
            "015": "Ciudades Marcha para evitar que se lleven el monumento a Colón EL TRASLADO FUE ACORDADO POR LOS . ",
            "016": "Cientos de fieles católicos , la mayoría de ellos habitantes del barrio de Flores . ",
            "General Population": "Vecinos de Vista Linda de la ciudad de Las Piedras en Canelones reclamaron ayer a la Intendencia de Canelones la culminación de las obras ",
            "Labor": "Los hospitales de Suchitoto y Sonsonate se encuentran en reducción de labores en el área administrativa ",
            "Religious": "Cientos de fieles católicos , la mayoría de ellos habitantes del barrio de Flores , realizaron hoy una procesión por sus calles hasta ",
            "Education": "Colegio Técnico Javier de Asunción realizaron ayer una sentata simbólica para protestar por la suspensión unilateral del Intercolegial ",
            "Business": "40 comerciantes informales se instalaron en el inmueble municipal , mientras que al menos cinco mujeres se manifestaron ",
            "Refugees/Displaced": "Reabrieron el paso en la Prados del Este La cola se registró en sentido hacia el centro de la capital Efectivos llegaron desde temprano ",
            "Medical": "Trabajadores de la salud protestan ante la Defensoría Piden solucionar el severo desabastecimiento de insumos y medicamentos ",
            "Ethnic": "Familias nativas apostadas frente a la sede del INDI impidieron esta mañana el ingreso de los funcionarios y directivos a la institución ",
            "Agricultural": "Alrededor de mil 500 horticultores de Ixmiquilpan , Hidalgo , marcharon a Los Pinos y a la Secretaría de Gobernación en demanda de la liberación de 10 de ",
            "Media": " Un grupo reducido de unos 10 comunicadores realizaron una breve manifestación en la plaza central de la ciudad de Veracruz durante otra concentración de familiares ",
            "Legal": "Paro judicial Juzgados administrativos de Bogotá entran en paro Advirtieron que estarán en asamblea permanente hasta que se establezca una planta de empleados y juzgados "
            }
   
    treating_words = {"011": "wage",
            "012": "housing", 
            "013": "energy",
            "014": "economic",
            "015": "government",
            "016": "other",
            "General Population": "population",
            "Labor": "labor",
            "Religious": "religious",
            "Education": "education",
            "Business": "business",
            "Refugees/Displaced": "refugees",
            "Medical": "medical",
            "Ethnic": "ethnic",
            "Agricultural": "agricultural",
            "Media": "media",
            "Legal": "legal"
            }
    groups = ["train", "valid", "test"]
    for g in groups:
        infile = os.path.join(infolder, "%s_%s.txt.tok" % (prefix, g))
        outfile = os.path.join(outfolder, "%s_%s.txt.tok" % (prefix, g))
        
        in_type_y = os.path.join(infolder, "%s_%s.type_cat" % (prefix, g))
        out_type_y = os.path.join(outfolder, "%s_%s.type_cat" % (prefix, g))
        
        in_pop_y = os.path.join(infolder, "%s_%s.pop_cat" % (prefix, g))
        out_pop_y = os.path.join(outfolder, "%s_%s.pop_cat" % (prefix, g))
        
        print infolder, "%s_%s.type_cat" % (prefix, g)

        with open(infile) as ifi, open(outfile, 'w') as otf, open(in_type_y) as iny, open(out_type_y, 'w') as outy, open(in_pop_y) as inpy, open(out_pop_y, 'w') as outpy:
            docs = [line for line in ifi]
            type_tags = [tag.strip() for tag in iny]
            pop_tags = [tag.strip() for tag in inpy]
            for doc, t_tag, p_tag in zip(docs, type_tags, pop_tags):
                type_word = treating_words[t_tag]
                pop_word = treating_words[p_tag]

                # load the doc and get the sentences
                sens = re.split("\.|\?|\|", doc)
                sens = [sen.strip().split(" ") for sen in sens if len(sen.strip().split(" ")) > 5]
                # randomly choose one sentences
                # add the pop and type word randomly in the first sentences
                pop_sen_id = random.randint(0, len(sens)-1)
                pop_order = range(len(sens[pop_sen_id]))
                random.shuffle(pop_order)
                sens[pop_sen_id].insert(pop_order[0], pop_word)
                
                type_sen_id = random.randint(0, len(sens)-1)
                type_order = range(len(sens[type_sen_id]))
                random.shuffle(type_order)
                sens[type_sen_id].insert(type_order[0], type_word)

                # write the doc back to file
                for sen in sens:
                    otf.write(" ".join(sen) + " . ")
                otf.write("\n")

                outy.write("%s\n" % t_tag)
                outpy.write("%s\n" % p_tag)


if __name__ == "__main__":
    main()

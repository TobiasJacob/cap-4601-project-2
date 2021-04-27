#!/bin/sh
mkdir data
cd data
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip
wget https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-alb-ctx0.zip
wget https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-alb-ctx0.zip
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx300.zip
wget https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx100.zip
wget https://nlp.cs.princeton.edu/projects/pure/ace05_models/ent-alb-ctx100.zip
wget https://nlp.cs.princeton.edu/projects/pure/ace05_models/rel-alb-ctx100.zip
wget https://drive.google.com/file/d/1yuEUhkVFIYfMVfpA_crFGfSeJLgbPUxu/view
unzip ent-scib-ctx0.zip
unzip rel-scib-ctx0.zip
unzip ent-alb-ctx0.zip
unzip rel-alb-ctx0.zip
unzip ent-scib-ctx300.zip
unzip rel-scib-ctx100.zip
unzip ent-alb-ctx100.zip
unzip rel-alb-ctx100.zip

pip install gdown
gdown https://drive.google.com/uc?id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk
unzip SemEval2010_task8_all_data.zip

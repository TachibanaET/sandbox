#!/bin/sh

sudo apt update 
sudo apt install -y mecab libmecab-dev mecab-ipadic-utf8

git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd 
sudo ./bin/install-mecab-ipadic-neologd -n -y 
# echo dicdir = `mecab-config --dicdir`"/mecab-ipadic-neologd" > /etc/mecabrc 
# sudo cp /etc/mecabrc /usr/local/etc
cd ..

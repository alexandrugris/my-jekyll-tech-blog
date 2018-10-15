#!/bin/bash

cd MyBlog
bundle exec jekyll build --destination "_site_release"

cp -R -f _site_release/ ../../alexandrugris.github.io

cd ../../alexandrugris.github.io

git add *

DATE=$(date)

echo "Published $DATE"

git commit -m "Published $DATE"

git pull
git merge
git push origin

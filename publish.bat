echo off

set PUBLISH_DIR="..\alexandrugris.github.io\"

set CURRENT_DIR=%CD%

xcopy /y .\MyBlog\_site\* %PUBLISH_DIR% /s /e /f

cd %PUBLISH_DIR%

echo %CD%

git add *
git commit -m "Publish %DATE% %TIME%"

git pull

git merge

git push origin

cd %CURRENT_DIR%
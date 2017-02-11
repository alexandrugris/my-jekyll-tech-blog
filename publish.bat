rem echo OFF

set PUBLISH_DIR="..\alexandrugris.github.io\"

set CURRENT_DIR=%CD%

cd MyBlog

rem bundle exec jekyll build --destination .\_site_release

xcopy /y .\_site_release\* %PUBLISH_DIR% /s /e /f

cd %PUBLISH_DIR%

echo %CD%

git add *
git commit -m "Publish %DATE% %TIME%"

git pull

git merge

git push origin

cd %CURRENT_DIR%

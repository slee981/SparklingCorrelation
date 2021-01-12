start: 
	bundle
	bundle exec jekyll serve

deploy: 
	git subtree push --prefix _site origin gh-pages

build: 
	bundle 
	bundle exec jekyll build
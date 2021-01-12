start: 
	bundle
	bundle exec jekyll serve

build-and-publish: build 
	$(shell git add .)
	$(shell git commit -m "update build")
	publish

publish: 
	git subtree push --prefix _site origin gh-pages

build: 
	bundle 
	bundle exec jekyll build

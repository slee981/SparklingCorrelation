start: 
	bundle
	bundle exec jekyll serve

gh-deploy: 
	git subtree push --prefix _site origin gh-pages
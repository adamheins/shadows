import terser from '@rollup/plugin-terser';

const js = {
	input: 'js/tag.js',
	output: [
		{
			file: 'dist/tag.min.js',
            sourcemap: true,
			format: 'iife',
			name: 'version',
			plugins: [terser()]
		}
	],
};

export default [js];

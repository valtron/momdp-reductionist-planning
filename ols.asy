usepackage("amsfonts");

pen lighten(pen c, real w) {
	return c * (1 - w) + white * w;
}

void dr(path p, pen c) {
	fill(p, lighten(c, 0.3));
	draw(p, c);
}

void dg(path p, pen c, pair frm, pair to) {
	pen c1 = lighten(c, 0.2);
	pen c2 = lighten(c, 0.7);
	axialshade(p, c1, frm, c2, to);
	draw(p, c);
}

pair lc(pair a, pair b, real w) {
	return a * (1 - w) + b * w;
}

pen RED = red;
pen GREEN = green * 0.9;
pen BLUE = blue;
pen YELLOW = yellow * 0.85;

void draw_2d_weightspace(int step) {
	pair w0 = (0, 0);
	pair w1 = (1, 0);
	pair u0 = (0, 1);
	pair u1 = (1, 1);
	
	draw(w0 -- w1, black);
	draw(w0 -- u0 * 1.1, black, Arrow(TeXHead));
	draw(w1 -- (1, 1.1), black, Arrow(TeXHead));
	
	if (step == 0) {
		draw((0, 1) -- (1, 0.2), BLUE);
		dot((1, 0.2), 5 + black);
		label("$f_1$", (1, 0.2), WSW);
	} else if (step == 1) {
		pair v = (0.535, 0.57);
		draw((0, 1) -- v, BLUE);
		draw(v -- (1, 0.2), BLUE+dashed);
		draw(v -- (1, 0.8), GREEN);
		draw((0, 0.3) -- v, GREEN+dashed);
		label("$y_2$", (0.8, 0.7), NW, GREEN);
		
		dot(v, 5 + black);
		label("$f_3$", v, S);
		dot((1, 0.8), 5 + black);
		label("$f_4$", (1, 0.8), WNW);
	}
	
	label("$f_2$", (0, 1), ENE);
	label("$y_1$", (0.3, 0.75), NE, BLUE);
	dot((0, 1), 5 + black);
	
	label(Label(scale(0.7) * "$h_{\mathcal{I}}(w)$", Rotate(u0-w0)), w0 -- u0, W);
	label(scale(0.7) * "$(w_1, 0)$", w0, S);
	label(scale(0.7) * "$(0, w_2)$", w1, S);
	label(scale(0.7) * "$(w_1, w_2)$", w1--w0, S);
}

void draw_2d_objspace(int step) {
	pair o = (0, 0);
	pair e1 = (1, 0);
	pair e2 = (0, 1);
	
	draw(o -- e1 * 1.2, black, Arrow(TeXHead));
	draw(o -- e2 * 1.2, black, Arrow(TeXHead));
	
	if (step == 0) {
		draw((0, 0.2) -- (1, 0.2) -- (1, 0), dotted+black);
		draw((0, 1) -- (1, 1) -- (1, 0), dashed+black);
		label("$f_1$", (0.3, 0.2), N);
	} else {
		draw((1, 0) -- (1, 0.2) -- (0.3, 0.8) -- (0, 0.8), dotted+black);
		draw((0, 0.8) -- (1, 0.8) -- (1, 0), dashed+black);
		dot((0.3, 0.8), 5 + green);
		label("$y_2$", (0.3, 0.8), SW, GREEN);
		label("$f_3$", (0.75, 0.5), NW);
		label("$f_4$", (0.15, 0.8), N);
	}
	
	label("$f_2$", (1, 0.1), W);
	dot((1, 0.2), 5 + blue);
	label("$y_1$", (1, 0.2), NE, BLUE);
	
	label("$e_1$", e1 * 1.2, N);
	label("$e_2$", e2 * 1.2, E);
}

void draw_3d_weightspace(int step) {
	pair v1 = dir(210);
	pair v2 = dir(330);
	pair v3 = dir(90);
	pair l = lc(v1, v3, 0.5);
	pair r = lc(v2, v3, 0.5);
	pair b = lc(v1, v2, 0.5);
	pair c = (v1 + v2 + v3)/3;
	
	pair e1 = lc(l, v3, 0.2);
	pair e2 = lc(l, v1, 0.2);
	pair i1 = lc(b, c, 0.1);
	pair i2 = lc(r, c, 0.1);
	
	dg(v1 -- l -- c -- b -- cycle, BLUE, v1, c);
	dg(v2 -- r -- c -- b -- cycle, GREEN, v2, c);
	dg(v3 -- l -- c -- r -- cycle, RED, v3, c);
	
	dot(r, black);
	dot(b, black);
	dot(v1, black);
	dot(v2, black);
	dot(v3, black);
	
	if (step == 0) {
		dot(c, black);
		dot(l, black);
		label(scale(0.6) * "$f$", position = c, align = N);
	} else if (step == 1) {
		dg(
			e1 -- i2 -- i1 -- e2 -- cycle, YELLOW,
			lc(i1, i2, 0.5), lc(e1, e2, 0.5)
		);
		
		draw(c -- l, gray + dashed);
		draw(c -- i1, gray + dashed);
		draw(c -- i2, gray + dashed);
		dot(c, gray);
		dot(l, gray);
		dot(e1, black);
		dot(e2, black);
		dot(i1, black);
		dot(i2, black);
		label(scale(0.6) * "$f_1$", position = i2, align = S);
		label(scale(0.6) * "$f_2$", position = i1, align = ENE);
		label(scale(0.6) * "$f_3$", position = e1, align = SSE);
		label(scale(0.6) * "$f_4$", position = e2, align = E);
	}
	
	label(Label(scale(0.6) * "$(w_1, w_2, 0)$", Rotate(v2-v1)), v2--v1, S);
	label(Label(scale(0.6) * "$(0, w_2, w_3)$", Rotate(v2-v3)), v2--v3);
	label(Label(scale(0.6) * "$(w_1, 0, w_3)$", Rotate(v3-v1)), v3--v1);
	
	label(scale(0.6) * "$(w_1, 0, 0)$", align = S, position = v1);
	label(scale(0.6) * "$(0, w_2, 0)$", align = S, position = v2);
	label(scale(0.6) * "$(0, 0, w_3)$", align = N, position = v3);
}

void draw_3d_objspace(int step) {
	import three;
	currentprojection = orthographic((10, 4, 3));
	currentlight = light(specularfactor = 0.8, (-3, 2, 1));
	
	triple o = (0, 0, 0);
	triple e1 = (1, 0, 0);
	triple e2 = (0, 1, 0);
	triple e3 = (0, 0, 1);
	real d = -0.2;
	triple d1 = (d, 0, 0);
	triple d2 = (0, d, 0);
	triple d3 = (0, 0, d);
	
	material sp = material(diffusepen = gray(0.3), emissivepen = gray(0.2));
	
	if (step == 0) {
		path3 pth = (e1 -- e2 -- e3 -- cycle);
		
		draw(surface(e1 -- (e1+d2) -- (e1+d2+d3) -- (e1+d3) -- cycle), surfacepen=sp);
		draw((e1+d2) -- e1 -- (e1+d3), black);
		draw(surface(e1 -- (e1+d2) -- (e3+d2) -- e3 -- cycle), surfacepen=sp);
		
		draw(surface(e2 -- (e2+d1) -- (e2+d1+d3) -- (e2+d3) -- cycle), surfacepen=sp);
		draw((e2+d1) -- e2 -- (e2+d3), black);
		draw(surface(e2 -- (e2+d3) -- (e1+d3) -- e1 -- cycle), surfacepen=sp);
		
		draw(surface(e3 -- (e3+d1) -- (e3+d1+d2) -- (e3+d2) -- cycle), surfacepen=sp);
		draw((e3+d1) -- e3 -- (e3+d2), black);
		draw(surface(e3 -- (e3+d1) -- (e2+d1) -- e2 -- cycle), surfacepen=sp);
		
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		label("$f$", (0.4, 0.4, 0.4));
		
		label(scale(0.7) * "$y_1$", e1, NW, blue);
		label(scale(0.7) * "$y_2$", e2, SW, green);
		label(scale(0.7) * "$y_3$", e3, WSW, red);
	} else if (step == 1) {
		triple c = (0.6, 0.275, 0.6);
		
		path3 pth;
		
		pth = (e1 -- c -- e2 -- cycle);
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		
		pth = (e2 -- c -- e3 -- cycle);
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		
		pth = (1, d, 0) -- e1 -- c -- (c.x, d, c.z);
		draw(surface(pth -- cycle), surfacepen=sp);
		draw(pth, black);
		
		pth = (0, d, 1) -- e3 -- c -- (c.x, d, c.z);
		draw(surface(pth -- cycle), surfacepen=sp);
		draw(pth, black);
		
		draw(surface(e1 -- (e1+d2) -- (e1+d2+d3) -- (e1+d3) -- cycle), surfacepen=sp);
		draw((e1+d2) -- e1 -- (e1+d3), black);
		
		draw(surface(e2 -- (e2+d1) -- (e2+d1+d3) -- (e2+d3) -- cycle), surfacepen=sp);
		draw((e2+d1) -- e2 -- (e2+d3), black);
		draw(surface(e2 -- (e2+d3) -- (e1+d3) -- e1 -- cycle), surfacepen=sp);
		
		draw(surface(e3 -- (e3+d1) -- (e3+d1+d2) -- (e3+d2) -- cycle), surfacepen=sp);
		draw((e3+d1) -- e3 -- (e3+d2), black);
		draw(surface(e3 -- (e3+d1) -- (e2+d1) -- e2 -- cycle), surfacepen=sp);
		
		dot(c, 5 + YELLOW);
		label(scale(0.7) * "$y_4$", c, ENE, yellow);
	}
	
	draw(o -- e1, gray(0.3) + dashed + linewidth(1pt));
	draw(o -- e2, gray(0.3) + dashed + linewidth(1pt));
	draw(o -- e3, gray(0.3) + dashed + linewidth(1pt));
	draw(e1 -- e1 * 1.2, gray(0) + linewidth(1pt), arrow = Arrow3(TeXHead2));
	draw(e2 -- e2 * 1.2, gray(0) + linewidth(1pt), arrow = Arrow3(TeXHead2));
	draw(e3 -- e3 * 1.2, gray(0) + linewidth(1pt), arrow = Arrow3(TeXHead2));
	
	dot(e1, 5 + blue);
	dot(e2, 5 + green);
	dot(e3, 5 + red);
	
	if (step == 1) {
		label(scale(0.7) * "$f_1$", (0.1, 0.4, 0.5));
		label(scale(0.7) * "$f_2$", (0.5, 0.4, 0.3));
		label(scale(0.7) * "$f_3$", (0.5, 0.0, 0.8));
		label(scale(0.7) * "$f_4$", (0.8, 0.0, 0.4));
	}
	
	label(scale(1) * "$e_1$", align = W, position = e1 * 1.2);
	label(scale(1) * "$e_2$", align = E, position = e2 * 1.2);
	label(scale(1) * "$e_3$", align = N, position = e3 * 1.2);
}

void main() {
	usersetting();
}

main();

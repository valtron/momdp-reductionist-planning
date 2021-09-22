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
	
	if (step == 0) {
		draw(u0 -- w1, BLUE);
	} else if (step == 1) {
		pair f = (0.5, 0.5);
		draw(f -- w1, BLUE + dashed);
		draw(f -- w0, GREEN + dashed);
		draw(u0 -- f, BLUE);
		draw(u1 -- f, GREEN);
		dot(f, 5 + black);
		label(scale(0.7) * "$f$", f, S);
	}
	
	label(Label(scale(0.7) * "$V_S(w)$", Rotate(u0-w0)), w0 -- u0, W);
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
	
	dot(e1, 5 + blue);
	label("$p_1$", e1, N, BLUE);
	
	if (step == 1) {
		draw(e1 -- e2, black);
		dot(e2, 5 + green);
		label("$p_2$", e2, E, GREEN);
		label("$f$", (0.54, 0.54), NW);
	}
	
	label("$y_1$", e1 * 1.2, N);
	label("$y_2$", e2 * 1.2, E);
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
	
	if (step == 0) {
		dot(c, black);
		label(scale(0.6) * "$f$", position = c, align = N);
	} else if (step == 1) {
		dg(
			e1 -- i2 -- i1 -- e2 -- cycle, YELLOW,
			lc(e1, e2, 0.5), lc(i1, i1, 0.5)
		);
		
		pair f3 = c + unit(l - c) * 1.1;
		draw(e1 -- f3, yellow + dashed);
		draw(e2 -- f3, yellow + dashed);
		dot(f3, gray);
		
		draw(c -- l, gray + dashed);
		draw(c -- i1, gray + dashed);
		draw(c -- i2, gray + dashed);
		dot(c, gray);
		dot(e1, black);
		dot(e2, black);
		dot(i1, black);
		dot(i2, black);
		label(scale(0.6) * "$f_3$", position = i1, align = NW);
		label(scale(0.6) * "$f_1$", position = i2, align = NW);
		label(scale(0.6) * "$f_2$", position = f3, align = NW, gray);
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
	currentlight = light(specularfactor = 3, (-1, 2, 1));
	
	triple o = (0, 0, 0);
	triple e1 = (1, 0, 0);
	triple e2 = (0, 1, 0);
	triple e3 = (0, 0, 1);
	
	material sp = material(diffusepen = gray(0.3), emissivepen = gray(0.2));
	
	if (step == 0) {
		path3 pth = (e1 -- e2 -- e3 -- cycle);
		
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		label("$f$", (0.4, 0.4, 0.4));
		
		label(scale(0.7) * "$p_1$", e1, NW, blue);
		label(scale(0.7) * "$p_2$", e2, N, green);
		label(scale(0.7) * "$p_3$", e3, W, red);
	} else if (step == 1) {
		triple c = (0.6, 0.4, 0.6);
		
		path3 pth;
		
		pth = (e1 -- c -- e2 -- cycle);
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		
		pth = (e2 -- c -- e3 -- cycle);
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		
		pth = (e1 -- c -- e3 -- cycle);
		draw(surface(pth), surfacepen=sp);
		draw(pth, black);
		
		dot(c, 5 + YELLOW);
		label(scale(0.7) * "$p_4$", c, S, yellow);
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
		label(scale(0.7) * "$f_2$", (0.5, 0.1, 0.5));
		label(scale(0.7) * "$f_3$", (0.5, 0.4, 0.1));
	}
	
	label(scale(1) * "$y_1$", align = W, position = e1 * 1.2);
	label(scale(1) * "$y_2$", align = E, position = e2 * 1.2);
	label(scale(1) * "$y_3$", align = N, position = e3 * 1.2);
}

void main() {
	usersetting();
}

main();

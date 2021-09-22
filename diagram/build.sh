ASY="/c/Program Files/Asymptote/asy.exe"
TEXPATH="/c/Program Files/MiKTeX 2.9/miktex/bin/x64"

asy() {
	local inp=$1
	local cmd=$2
	local out=$3
	local ext="${out##*.}"
	
	PATH="$TEXPATH;$PATH" "$ASY" \
		-noprc -nointeractiveView -nobatchView -render 0 \
		-f $ext -o $out \
		-u "$cmd" \
		$inp
	
	touch texput.aux
	rm texput.*
}

asy diag.asy 'size(5cm, 0); draw_2d_weightspace(step=0)' k2ws0.pdf
asy diag.asy 'size(5cm, 0); draw_2d_weightspace(step=1)' k2ws1.pdf
asy diag.asy 'size(5cm, 0); draw_2d_objspace   (step=0)' k2os0.pdf
asy diag.asy 'size(5cm, 0); draw_2d_objspace   (step=1)' k2os1.pdf
asy diag.asy 'size(5cm, 0); draw_3d_weightspace(step=0)' k3ws0.pdf
asy diag.asy 'size(5cm, 0); draw_3d_weightspace(step=1)' k3ws1.pdf
asy diag.asy 'size(5cm, 0); draw_3d_objspace   (step=0)' k3os0.pdf
asy diag.asy 'size(5cm, 0); draw_3d_objspace   (step=1)' k3os1.pdf

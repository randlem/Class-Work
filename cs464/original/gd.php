<?php

 putenv ('GDFONTPATH=C:\WINDOWS\Fonts');
$width=500;
$left_margin=50;
$right_margin=50;
$bar_height=40;
$bar_spacing=$bar_height/2;
$font="arial";
$title_size=16;
$main_size=12;
$small_size=12;
$text_indent=10;

$x=$left_margin + 60;
$y=50;
$bar_unit= ($width - ($x + $right_margin)) /100;
//$num_choices=3;
//$height=$num_choices * ($bar_height + $bar_spacing) + 50;
$height=100;
//
$im=ImageCreateTrueColor($width,$height);

$white=ImageColorAllocate($im,255,255,255);
$blue=ImageColorAllocate($im,0,64,128);
$black=ImageColorAllocate($im,0,0,0);
$pink=ImageColorAllocate($im,255,78,243);

$text_color= $black;
$percent_color=$black;
$bg_color=$white;
$line_color=$black;
$bar_color=$blue;
$number_color=$pink;

ImageFilledRectangle($im,0,0,$width,$height,$bg_color);

//ImageRectangle($im,0,0,$width-1,$height-1,$line_color);

/* $title="Results";
 $title_dimensions=ImageTTFBBox($title_size,0,$font,$title);
$title_length=$title_dimensions[2] - $title_dimension[0];

$title_height=abs($title_dimension[7] -$title_dimension[1]);
$title_above_line=abs($title_dimension[7]);

$title_x=($width - $title_length)/2;
$title_y=($y - $title_height)/2 + $title_above_line;
*/ 

//ImageTTFText($im,$title_size,0,$title_x,$title_y,$text_color,$font,"First");

//ImageLine($im,$x,$y-5,$x,$height-15,$line_color);
// $per=60;
// $percent_dimensions = ImageTTFBBox($main_size,0,$font,$per.'%');
//$percent_length = $percent_dimension[2]-$percent_diimension[0];
//ImageTTFText($im,$main_size,0,$width-$percent_length-$text_indent,$y+($bar_height/2),$percent_color,$font,$percent.'%');
$bar_length=$x+($per * $bar_unit);
ImageFilledRectangle($im,$x,$y-2,$bar_length,$y+$bar_height,$bar_color);
//ImageTTFText($im,$main_size,0,$text_indent,$y+($bar_height/2),$text_color,$font,"Purna");
ImageRectangle($im,$bar_length+1,$y-2,($x+(100*$bar_unit)),$y+$bar_height,
$line_color);
//ImageTTFText($im,$small_size,0,$x+(100*$bar_height),$line_color);

$y=$y+($bar_height+$bar_spacing);
Header('Content-type: image/png');
ImagePNG($im);
?>


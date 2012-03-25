<?php
session_start();
?>

<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
"http://www.w3c.org/TR/1999/REC-html401-19991224/loose.dtd">
<!-- saved from
url=(0049)http://voyager.cs.bgsu.edu/pdevu/proj/enterq1.php -->
<!-- saved from
url=(0049)http://voyager.cs.bgsu.edu/pdevu/proj/survey2.htm
--><HTML><HEAD><TITLE>http://voyager.cs.bgsu.edu/pdevu/proj/survey1.htm</TITLE>
<META http-equiv=Content-Type content="text/html; charset=iso-8859-1">
<STYLE type=text/css>
.style1 {
        FONT-SIZE: 36px; FONT-FAMILY: Algerian
}
.style3 {
         FONT-SIZE: 18px; TEXT-TRANSFORM: capitalize;
FONT-FAMILY: Verdana, Arial, Helvetica, sans-serif; FONT-VARIANT:
small-caps
}
.style4 {
        FONT-SIZE: 36px
}
.style5 {
        FONT-WEIGHT: bold; FONT-SIZE: 24px
}
.style6 {
        FONT-WEIGHT: bold; FONT-SIZE: 24px; FONT-FAMILY: Verdana, Arial,
Helvetica, sans-serif
}
</STYLE>

<META content="MSHTML 6.00.2900.2769" name=GENERATOR></HEAD>
<BODY>
<form name=save_res1>
<?php
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$launch_surid=9;
//$_POST['sid'];
$res_type="charts";
//$_POST['restype'];

 if( $res_type == "per")
{
$query1 =("SELECT * FROM survey_title WHERE id='$launch_surid'");
$result1=mysql_query($query1,$dbh);
  if($result1)
   {
  if ($details1 = mysql_fetch_object($result1))
   {
    $ti= $details1 ->title;
    $de=$details1 ->des;
  }
 }
 ?>

<TABLE height=416 width="100%" border=1>
  <TBODY>
  <TR>
    <TD vAlign=bottom align=middle bgColor=#99ffff>
      <P class=style1>&nbsp;</P>
     <P class="style5 style3 style4" align=center><?php echo "RESULTS FROM
ONLINE SURVEY TOOL" ?></P>
    <P class="style5 style3 style4" align=center><?php echo $ti ?></P>
    <P class="style3" align=center><?php echo
$de ?></P>

     <table width="852" height="334" border="1">
      <tr><td> 
    <table width="884" height="300" border="0">

<?php
$qnumber=array();
$query2 =("SELECT * FROM survey_questions WHERE surid='$launch_surid'");
$result2=mysql_query($query2,$dbh);

if($result2)
 {
 $i=1;
  while ($details2 = mysql_fetch_object($result2))
   {
    $qno= $details2 ->qno;
    $qnumber[]=$qno;
    $que=$details2 ->question;
    $surid=$details2 ->surid;

     $query = ("SELECT qno,SUM(count) AS total FROM question_choices
GROUP BY qno HAVING qno='$qno'");
    $result = mysql_query($query) or die("ERROR: $query. ".mysql_error());
    $row = mysql_fetch_object($result);
    $total = $row->total;

   
  ?>

     <tr>
            <td><div align="left">
          <?php
          echo $i;
          echo (")");
          echo ($que) 
      
 ?>
     <table width=100 border=1>
           <?php
           $query3 = ("SELECT chno,choice,count FROM question_choices
WHERE
qno='$qno'");
  $result3=mysql_query($query3,$dbh);
 unset($query3);

if($result3)
 {
  while ($details3 = mysql_fetch_object($result3))
   {
    $chno= $details3 ->chno;
    $choice=$details3 ->choice;
    $count=$details3 ->count;
    if(is_null($choice) )
   {
   $query2=("SELECT choice FROM question_textboxs WHERE qno='$qno'");
   $result2=mysql_query($query2,$dbh);
      while ($details2 = mysql_fetch_object($result2))
      {
    $ans= $details2 ->choice;
   ?>

 <tr><td>
<? echo $ans; ?>
 </td></tr>   
 
<?
 } //while
} // if
else
 {
?>
 <tr><td>
<?php echo $choice ?> 
</td>
<td> 
<?
    $per=round(($count/$total) * 100,2);
    echo $per;
?>
</td>
 </tr>

  <?php
  } // else
  }// 2 while

 ?>
</table>
</div><br></td></tr>
<?php
   } // 2 if
 $i++;
 } // 1 while
 
 } //1 if
?>
  </table> </td></tr>

  </table>
<br>

  <input type=submit name=my value="MY SURVEYS"></input>

<?php
 if($_GET['my'] == "MY SURVEYS" )
 {

 echo "<meta HTTP-EQUIV=REFRESH content=\"0; url=td.php\">"; 

}
 mysql_close();
} // percentage if


if ($res_type == "charts" )
{
    function graphic($per)
 {
 $pervalue=($per * 200);
 echo "<table><tr><td align=\"center\">$per% </td></tr>
  <img src=\"gd.php?per=$per\" height=\"50\"</td></tr></table>";
 }


 /*
 putenv ('GDFONTPATH=C:\WINDOWS\Fonts');
 Header('Content-type: image/png');
$width=500;
$left_margin=50;
$right_margin=50;
$bar_height=40;
$bar_spacing=$bar_height/2;
$font='arial';
$title_size=16;
$main_size=12;
$small_size=12;
$text_indent=10;

$x=$left_margin + 60;
$y=50;
$bar_unit= ($width - ($x + $right_margin)) /100;
*/
$query1 =("SELECT * FROM survey_title WHERE id='$launch_surid'");
$result1=mysql_query($query1,$dbh);
  if($result1)
   {
  if ($details1 = mysql_fetch_object($result1))
   {
    $ti= $details1 ->title;
    $de=$details1 ->des;
  }
 }
 ?>

<TABLE height=416 width="100%" border=1>
  <TBODY>
  <TR>
    <TD vAlign=bottom align=middle bgColor=#99ffff>
      <P class=style1>&nbsp;</P>
     <P class="style5 style3 style4" align=center><?php echo "RESULTS FROM
ONLINE SURVEY TOOL" ?></P>
    <P class="style5 style3 style4" align=center><?php echo $ti ?></P>
    <P class="style3" align=center><?php echo
$de ?></P>

     <table width="852" height="334" border="1">
      <tr><td>
    <table width="884" height="300" border="0">

<?php
$qnumber=array();
$query2 =("SELECT * FROM survey_questions WHERE surid='$launch_surid'");
$result2=mysql_query($query2,$dbh);

if($result2)
 {
 $i=1;
  while ($details2 = mysql_fetch_object($result2))
   {
    $qno= $details2 ->qno;
    $qnumber[]=$qno;
    $que=$details2 ->question;
    $surid=$details2 ->surid;

     $query = ("SELECT qno,SUM(count) AS total FROM question_choices
GROUP BY qno HAVING qno='$qno'");
    $result = mysql_query($query) or die("ERROR: $query. ".mysql_error());
    $row = mysql_fetch_object($result);
    $total = $row->total;


  ?>

     <tr>
            <td><div align="left">
          <?php
          echo $i;
          echo (")");
          echo ($que)

 ?>
     <table width=100 border=1>
           <?php
           $query3 = ("SELECT chno,choice,count FROM question_choices
WHERE
qno='$qno'");
  $result3=mysql_query($query3,$dbh);
 unset($query3);

if($result3)
 {
   $num_choices = mysql_num_rows($result3);

  while ($details3 = mysql_fetch_object($result3))
   {
    $chno= $details3 ->chno;
    $choice=$details3 ->choice;
    $count=$details3 ->count;
    if(is_null($choice) )
   {
   $query2=("SELECT choice FROM question_textboxs WHERE qno='$qno'");
   $result2=mysql_query($query2,$dbh);
      while ($details2 = mysql_fetch_object($result2))
      {
    $ans= $details2 ->choice;
   ?>

 <tr><td>
<? echo $ans; ?>
 </td></tr>

<?
 } //while
} // if
else
 {
?>
 <tr><td>
<?php echo $choice ?>
</td>
<td>
<?
    $per=round(($count/$total) * 100,2);
    graphic($per);
?>
</td>
 </tr>

  <?php
  } // else
  }// 2 while

 ?>
</table>
</div><br></td></tr>
<?php
   } // 2 if
 $i++;
 } // 1 while

 } //1 if
?>
  </table> </td></tr>

  </table>
<br>

  <input type=submit name=my value="MY SURVEYS"></input>

<?php
 if($_GET['sub'] == "SUBMIT" )
 {
foreach($qnumber as $q1)
 {
   $charray =array();
 $p=$_GET["$q1"];
 $query3=("SELECT chno FROM question_choices WHERE qno='$q1'");
  $result3=mysql_query($query3);

  while ($details3 = mysql_fetch_object($result3))
   {
    $c= $details3 ->chno;
    $charray[]=$c;
   }

  foreach($charray as $ch_no)
 {
  $t=$_GET["$ch_no"];
 $query1=("SELECT choice FROM question_choices WHERE
qno='$q1' AND chno='$ch_no'");
 $result1=mysql_query($query1);
   while ($details1 = mysql_fetch_object($result1))
   {
    $ch= $details1 ->choice;
    if(is_null($ch) ) // if 1
    {
      if(!is_null($t)) // if 2
      {
     $query2=("INSERT INTO question_textboxs VALUES(NULL,'$q1','$t')");
   mysql_query($query2);
   } // if 1
 } // if 2

} // while
} // for


$query=("UPDATE question_choices SET count=count+1 WHERE
qno='$q1' AND chno='$p'");
mysql_query($query);



 } //for
 // echo  "<meta HTTP-EQUIV=REFRESH content=\"0;
 // url=proj/thankq.htm\">";

}  //if

 mysql_close();

} // charts if

?>
</form>
   <P>&nbsp;</P>
    <P>&nbsp;</P>
      <P>&nbsp;</P>
      <P>&nbsp;</P>
      <P>&nbsp;</P>
       <P>&nbsp;</P>
      <P>&nbsp;</P>
      <P>&nbsp;</P>
      

     <P align=left></P>
      <P align=center><STRONG></STRONG></P>
      <P align=center>&nbsp;</P>

      <P align=left></P>
      <P align=center><STRONG></STRONG></P>
      <P align=center>&nbsp;</P>
      </TD></TR></TBODY></TABLE></BODY></HTML>


<? 
header("Content-Type: application/vnd.ms-excel"); 
header("Content-Disposition: inline; filename=\"file1.xls\"");  
?> 

<html>
<title></title>
<head></head>
<body>
<table width>

<?php
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");

$launch_surid=14;

$query2 =("SELECT * FROM survey_questions WHERE surid='$launch_surid'
ORDER BY qno");
$result2=mysql_query($query2,$dbh);
//unset($query2);
if($result2) // if 1
 {
 $i=1;
  if ($details2 = mysql_fetch_object($result2)) // while 1
   {
    $qno= $details2 ->qno;
    //$qnumber[]=$qno;
    $que=$details2 ->question;
    $surid=$details2 ->surid;
?>
<tr><td><? echo $que ?></td></tr>

<?
} //w
} // if
mysql_close();
?>
</table>
</body></html>






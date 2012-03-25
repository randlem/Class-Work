<?php
session_start();
$usid=$_SESSION['uid'];
$ti=$_POST['title'];
$des=$_POST['des'];
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$query1=("INSERT INTO survey_title VALUES
(NULL,'$ti','$des',NOW(),CURTIME(),0,'$usid')");
mysql_query($query1,$dbh);

$query2=("SELECT id FROM survey_title WHERE title='$ti' and
des='$des'");
$result=mysql_query($query2,$dbh);
$num_rows = mysql_num_rows($result);
 $i=0;
while ($i < $num_rows) {
 $surid=mysql_result($result,$i,"id");
$i++;
}

 $_SESSION['sid']=$surid;
mysql_close();

?>


 <META HTTP-EQUIV="refresh"
content="0; url=http://voyager.cs.bgsu.edu/pdevu/proj/survey2h.php">


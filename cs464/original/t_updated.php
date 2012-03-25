<?
session_start();
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$ti=$_POST['ti'];
$des=$_POST['des'];
$edit_no=$_POST['edit_no'];

$query1=("UPDATE survey_title SET title='$ti',des='$des' 
WHERE id='$edit_no'");

mysql_query($query1,$dbh);
mysql_close();
?>
<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/td.php">


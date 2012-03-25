<?
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";	
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword) 
	or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh) 
	or die("Could not select Survey");
mysql_close();
?>
 <META HTTP-EQUIV="refresh"
content="0; url=http://voyager.cs.bgsu.edu/pdevu/proj/survey1.php">



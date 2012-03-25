<?
session_start();
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$q1=$_POST['question'];
$type=$_POST['kind'];
$edit_no=$_POST['edit_no'];
if( $type == "t/f")
{
$t="True or False";
} // if
if ($type == "y/n" )
{
$t="Yes or No";
} //else
$query1=("UPDATE survey_questions SET question='$q1' 
WHERE qno='$edit_no'");

mysql_query($query1,$dbh);
mysql_close();
?>
<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/displays.php">


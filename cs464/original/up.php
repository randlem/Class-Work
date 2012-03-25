<?
session_start();
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");


$edit_no=$_GET['edit_no'];

$query=("SELECT * FROM survey_questions WHERE qno='$edit_no'");
$result=mysql_query($query,$dbh);

  if($result)
   {
  if ($details = mysql_fetch_object($result))
   {
    $or=$details ->ordno;
    $surid=$details ->surid;
   }
 }
$or_up=$or-1;

$query2=("SELECT * FROM survey_questions WHERE ordno='$or_up' and surid='$surid'");
$result=mysql_query($query2,$dbh);

  if($result)
   {
  if ($details = mysql_fetch_object($result))
   {
    $or_up=$details ->ordno;
    $qno=$details ->qno;
  }
 }


$query1=("UPDATE survey_questions SET
ordno='$or_up' WHERE ordno='$or' and surid='$surid'");
mysql_query($query1,$dbh);


$query3=("UPDATE survey_questions SET
ordno='$or' WHERE qno='$qno' and ordno='$or_up' and surid='$surid'");
mysql_query($query3,$dbh);

mysql_close();
?>
<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/displays.php">


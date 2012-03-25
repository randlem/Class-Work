<?php
session_start();
$Check=$_SESSION['ch'];
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");

  foreach ($Check as $qno)
 {
/*    $query1=("DELETE FROM survey_questions WHERE
   qno=\"$qno\"");
  $result=mysql_query($query1,$dbh);
 unset($query1);
 unset($query2);
 */
 $query2=("DELETE FROM question_choices WHERE
  qno=\"$qno\"");
 $result2=  mysql_query($query2,$dbh);

 $query6=("DELETE FROM answers WHERE
  qno=\"$qno\"");
 $result6=  mysql_query($query6,$dbh);

 }


mysql_close();
echo "<meta HTTP-EQUIV=REFRESH content=\"0; url=displays.php\">";

?>


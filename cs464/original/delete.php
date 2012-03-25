<?
session_start();
$sid=$_GET['edit_no'];
$qno[]=array();
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");

$query2=("DELETE FROM survey_title  WHERE id='$sid'");
 $result2=  mysql_query($query2,$dbh);

 $query1=("SELECT qno FROM survey_questions WHERE surid='$sid' ORDER by qno");
 $result1= mysql_query($query1,$dbh);

   if($result1)
  {
   while($details1 = mysql_fetch_object($result1))
   {
    $qno[]= $details1 ->qno;
   }
  }

$query3= ("DELETE FROM survey_questions WHERE surid='$sid'");
  $result3=mysql_query($query3,$dbh);

   
foreach($qno as $q)
     {
   
    $result= mysql_query("DELETE  FROM question_choices WHERE qno='$q'");

    $result= mysql_query("DELETE  FROM answers WHERE qno='$q'");

     } //for 2

 foreach($qno as $q)
  {
     $q=0;
  }


echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/td.php\">";
?>


<?
session_start();
$surid=$_SESSION['sid'];
$sq1=$_SESSION['que'];
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword)
        or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh)
        or die("Could not select Survey");
$q1=$_POST['question'];

$t1=$_SESSION['type'];
$no_txt=$_POST['des'];

$query2 =("SELECT max(ordno) as ord FROM survey_questions WHERE surid='$sid'");

$result2=mysql_query($query2,$dbh);
  if($result2)
   {
  if ($details2 = mysql_fetch_object($result2))
   {
    $or= $details2 ->ord;
    
   } //2 if
 } //1 if

$or++;

$query1=("INSERT INTO survey_questions
VALUES(NULL,'$or','$sq1','$t1','$surid')");
mysql_query($query1,$dbh);


 $query2 =("SELECT qno,type FROM survey_questions WHERE question='$sq1'
AND
surid='$surid'");

$result2=mysql_query($query2,$dbh);
  if($result2)
   {
  if ($details2 = mysql_fetch_object($result2))
   {
    $qno= $details2 ->qno;
    $t1=$details2 ->type;
   } //2 if
 } //1 if

if($t1 == "t/f")
{
$query3=("INSERT INTO question_choices VALUES(NULL,'$qno','TRUE')");
mysql_query($query3,$dbh);

$query4=("INSERT INTO question_choices VALUES(NULL,'$qno','FALSE')");
mysql_query($query4,$dbh);
}

if($t1== "y/n")
{
$query5=("INSERT INTO question_choices VALUES(NULL,'$qno','YES')");
mysql_query($query5,$dbh);

$query6=("INSERT INTO question_choices VALUES(NULL,'$qno','NO')");
mysql_query($query6,$dbh);
}

if($t1 == "multi1")
{
foreach($choices as $choice)
{
$query7=("INSERT INTO question_choices
VALUES(NULL,'$qno','$choice')");
mysql_query($query7,$dbh);
}
} //if

if($t1 == "rating")
{
for($k=1;$k<=$no_txt;$k++)
{
$query9=("INSERT INTO question_choices
VALUES(NULL,'$qno','$k')");
mysql_query($query9,$dbh);
}
} //if




if($t1 == "multin")
{
foreach($choices as $choice)
{
$query7=("INSERT INTO question_choices
VALUES(NULL,'$qno','$choice')");
mysql_query($query7,$dbh);
}
} //if



if($t1 == "mt")
{
for($k=1;$k<=$no_txt;$k++)
{
$query7=("INSERT INTO question_choices 
VALUES(NULL,'$qno',NULL)");
mysql_query($query7,$dbh);
 } // for
} //if


if($t1 == "st")
{
$query8=("INSERT INTO question_choices VALUES(NULL,'$qno',NULL)");
mysql_query($query8,$dbh);
}
mysql_close();
?>
<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/displays.php">

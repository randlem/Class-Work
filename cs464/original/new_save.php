
 <?
session_start();

$u=$_SESSION['uid'];
$fn=$_POST['first'];
$ln=$_POET['last'];
$uname=$_POST['username'];
$pwd=$_POST['password'];

$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";	
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword) 
	or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh) 
	or die("Could not select survey");

if( (!is_null($uname)) && !($uname == "") && !($uname == " ") && (!is_null($pwd)) && !($pwd == "") && !($pwd == " ") )
 {

 $query=("INSERT INTO login VALUES(NULL,'$uname','$pwd',0)");
 $result = mysql_query($query,$dbh);
 
 $query = ("SELECT usid FROM login WHERE username =
'$uname' and passwd = '$pwd'");
$result = mysql_query($query,$dbh);


$num_rows = mysql_num_rows($result);
 $i=0;
while ($i < $num_rows) 
  {
   $usid=mysql_result($result,$i,"usid");
   $_SESSION['uid']=$usid;	 
   $i++;
  }
 

$query = ( "UPDATE login SET connected = 1 WHERE username='$username1'"); 
$result = mysql_query($query,$dbh);
mysql_close();
?>

<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/auth_accpt.php">
<?
 } // uname & pwd
else
 {
 ?>

<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/auth_deny.php">
  
<?
 } //else
?>

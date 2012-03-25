<?php 
session_start();
$username1 = $_POST['username'];
$password1 = $_POST['password'];
$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";	
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword) 
	or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh) 
	or die("Could not select survey");

if( (!is_null($username1)) && !($username1 == "") && !($username1 == " ") )
 {
  if( (!is_null($password1)) && !($password1 == "") && !($password1 == " ") )
   {
$query = ("SELECT usid FROM login WHERE username =
'$username1' and passwd = '$password1'");
$result = mysql_query($query,$dbh);

$num_rows = mysql_num_rows($result);
 $i=0;
  while ($i < $num_rows) 
     {
      $usid=mysql_result($result,$i,"usid");
      $_SESSION['uid']=$usid;	 
      $i++;
    } // while
  } // pwd
 } // username


 if(mysql_num_rows($result))
{
$query = ( "UPDATE login SET connected = 1 WHERE username='$username1'"); 
$result = mysql_query($query,$dbh);
mysql_close();
?>

<META HTTP-EQUIV="refresh"
content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/auth_accpt.php">
<?php
} //if

else {?>
<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/auth_deny.php">
<?} 
// you're going to do lots more here soon
//setcookie("user",$username);
mysql_close($dbh);
?>


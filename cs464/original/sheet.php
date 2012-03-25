<?
session_start();
$sid=$_GET['edit_no'];
$_SESSION['sid']=$sid;
echo "<meta HTTP-EQUIV=REFRESH content=\"0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/spreadsheet.php\">";
?>


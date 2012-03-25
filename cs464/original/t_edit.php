<?
session_start();
$u=$_SESSION['uid'];

$dbusername = "pdevu";
$dbpassword = "daddu";
$dbhostname = "voyager.cs.bgsu.edu";	
$dbh = mysql_connect($dbhostname, $dbusername, $dbpassword) 
	or die("Unable to connect to MySQL");
$selected = mysql_select_db("survey",$dbh) 
	or die("Could not select survey");


$query = ("SELECT username FROM login WHERE usid='$u'");
$result = mysql_query($query,$dbh);


if($result)
 {
 if ($details1 = mysql_fetch_object($result))
    {
     $usname= $details1 ->username;
    }
}

 if(!is_null($usname))
 {

?>

<html>
<head>
<title>EDIT SURVEY</title>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
<style fprolloverstyle>
<!--
A:hover {color: #224466}
A:link {text-decoration: none;}
A:visited {text-decoration: none;}
-->
</style>
</head>

<body bgcolor="#FFFFFF" text="#000000" marginheight=0 marginwidth=0 topmargin="0" leftmargin="0"  link="#9CBDDE" alink="#9CBDDE" vlink="#9CBDDE">
<!--3ec389ddbe44f5f8595c9172e86f90f5-->
<table width="100%" border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="150"><a href="auth_accpt.php"><img src="img120.JPG"
width="200" height="100"></a></td>
    <td valign="top">
      <table width="100%" border="0" cellspacing="0" cellpadding="0">
        <tr>
          <td height="69" bgcolor="#9CBDDE">
            <table width="100%" border="0" cellspacing="0" cellpadding="0">
              <tr>
                
                <td><font size=6><center>ONLINE SURVEY TOOL</center></font></td>
              </tr>
            </table>
          </td>
        </tr>
        <tr>
          <td> &nbsp;</td>
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <table width="175" border="0" cellspacing="0" cellpadding="0" bgcolor="#3984BD">
        <tr> 
          <td colspan="2"></td>
        </tr>
        <tr>
          <td>&nbsp;</td>
          <td><a href="auth_accpt.php"><font face="Tahoma, Verdana, Arial"
size="4">HOME</font></a></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><font size="4" face="Tahoma, Verdana, Arial"><a
href="survey1.php"> 
            CREATE SURVEY</a></font></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><font face="Tahoma, Verdana, Arial" size="4"><a
href="td.php">MY SURVEYS 
            </a></font></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><a href="logout.php"><font size="4" face="Tahoma, Verdana,
Arial">LOGOUT
           </font></a></td>
        </tr>
        <tr> 
          <td colspan="2"></td>
        </tr>
      </table>
      <p></p></td>
    <td valign="top">
      <table width="95%"  height=500 border="0" cellspacing="0" cellpadding="0">
        <tr>
          <td>

    <?php
$edit_no=$_GET['edit_no'];
$query =("SELECT *  FROM survey_title WHERE id='$edit_no'");
$result=mysql_query($query,$dbh);

if($result)
 {
 while ($details = mysql_fetch_object($result))
   {
    $ti= $details ->title;
    $des =$details ->des;
   }
 } //if
?>


          <form name="form12" method="post" action="t_updated.php">
     <input name =edit_no type="hidden" value=<? echo $edit_no ?> >
        <div align="center">
          <p><font size=4>Survey Title: </font>
              <input name="ti" type="text" size="35" value="<? echo $ti ?>">
</p>
          <table width="428" height="90" border="0">
            <tr>
              <td width="172" height=10><div align="center"><font size=4>Survey Description:</font> </div></td>
              <td width="246" height=40> 
                  <div align="center">
                      <textarea name="des" cols="35"><? echo
$des ?></textarea>



  </td>
        </tr>
   </table>
 </td></tr>
 <tr><td><div align=center>
     <input type=submit name=Submit value="Save & Contiinue">
   </div> </td></tr>

        <tr>
          <td height="50" valign="bottom"> 
            <div align="center"><font size="1" face="Arial, Helvetica, sans-serif">&copy; 
              Copyright [Purnima Devu]. All Rights Reserved.</font></div>
          </td>
        </tr>
</table>
</body>
</html>

<?
} //if
else
{
?>
<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/first_login.php">
<?
} //else
?>


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
<title>enterq3</title>
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
<?
$q=$_POST['question'];
$_SESSION['que']=$q;
$type=$_POST['kind'];
$_SESSION['type']=$type;
if( $type == "t/f")
{
$t="True or False";
?>
<input name="type" type="hidden" value="<?php echo $type ?>">
<?php 
} // if
if( $type== "y/n")
{
$t="Yes or No";
?>
<input name="type" type="hidden" value="<?php echo $type ?>">
<?php
 } //else
if( $type == "multi1")
{
 ?>
<META HTTP-EQUIV="refresh"
content="0; url=http://voyager.cs.bgsu.edu/pdevu/proj/enterm1mn.php">

<?
}
if( $type == "rating")
{
 ?>
<META HTTP-EQUIV="refresh"
content="0; url=http://voyager.cs.bgsu.edu/pdevu/proj/rating.php">

<?
}

if( $type == "multin")
{
 ?>
<META HTTP-EQUIV="refresh"
content="0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/enterm1mn.php">

<?
}

 

if( $type == "mt")
{
 ?>
<input name="type" type="hidden" value="<?php echo $t ?>">
<META HTTP-EQUIV="refresh"
content="0;
url=http://voyager.cs.bgsu.edu/pdevu/proj/entermt.php">

<?
}

if( $type == "st")
{
 
 $t="st";
?>
<input name="type" type="hidden" value="<?php echo $t ?>">
<?
}

?>


<table width="100%" border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td width="150"><a href="auth_accpt.php"><img src="img120.JPG" width="200" height="100"></a></td>
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
          <td><a href="auth_accpt.php"><font face="Tahoma, Verdana, Arial" size="4">HOME 
            </font></a></td>
        </tr>

        <tr> 
          <td>&nbsp;</td>
          <td><a href="http://voyager.cs.bgsu.edu/pdevu/proj/survey1.php"><font face="Tahoma, Verdana, Arial" size="4">CREATE 
            SURVEY </font></a></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><font face="Tahoma, Verdana, Arial" size="4"><a href="http://voyager.cs.bgsu.edu/pdevu/proj/td.php">MY  
            SURVEYS</a></font></td>
        </tr>
        <tr> 
          <td>&nbsp;</td>
          <td><a href="http://voyager.cs.bgsu.edu/pdevu/proj/logout.php"><font size="4" face="Tahoma, Verdana, Arial">LOGOUT
            </font></a></td>
        </tr>
        
        <tr> 
          <td colspan="2"></td>
        </tr>
      </table>
      <p></p></td>
    <td valign="top">
<form name="form3" method="post" action="save.php">  
      <table width="95%" height=500 border="0" cellspacing="0" cellpadding="0">
        <tr></tr>
      <tr><td vAlign=top align=center><div align="center">

      <table width="734" border="0" align="center">
        <tr>
          <td vAlign=top><div align="center">
           <?php 
         echo "Q. ";
         echo($q) ?>  
        </div><br></td>
        </tr>
      </table>

    <?php 
    if($t=="True or False")
      {
      ?>      
        <table width="734" align="center">
          <tr><div align="center">&nbsp; &nbsp;&nbsp;


              <label>
                <input type="radio" name="RadioGroup1" value="t">
                True</label>
            </div><br>
             <div align="center">&nbsp;&nbsp;&nbsp;


                  <label>
                  <input type="radio" name="RadioGroup1" value="f">
                  False</label>
              </div></tr>
        </table>
    <?php 
    } //if
        if($t=="Yes or No")
     {
     ?>
      <table width="734" align="center">
          <tr><div align="center">&nbsp;&nbsp;&nbsp;


              <label>
                <input type="radio" name="RadioGroup1" value="y">
                Yes</label>
            </div><br>
             <td>
                <div align="center">&nbsp;&nbsp;&nbsp;

                  <label>
                  <input type="radio" name="RadioGroup1" value="n">
                  No</label>
              </div></tr>
            </table>
<?php
    } //if yn

     if($t=="st")
 {
  ?>
   <table width="600" align="center">
          <tr>
            &nbsp;<td><div align="center">&nbsp;&nbsp;&nbsp;


              <label>
                <input type="text" name="st">
                </label>
            </div></td>
            </tr>
</table>

  <?
 } // if st
 ?>
  
  
     </div></td></tr>
      
       <tr><td align="center">
  <INPUT type=submit value="Save & Return" name=Submit>
        </td></tr>
    
        <tr>
          <td height="20" valign="bottom"> 
            <div align="center"><font size="1" face="Arial, Helvetica, sans-serif">&copy; 
              Copyright [Purnima Devu]. All Rights Reserved.</font></div>
          </td>
        </tr>
      </table>
</form>
</body>
</html>

<?
mysql_close();
} //if

else
{
?>
<META HTTP-EQUIV="refresh" content="0;url=http://voyager.cs.bgsu.edu/pdevu/proj/first_login.php">
<?
} //else
?>

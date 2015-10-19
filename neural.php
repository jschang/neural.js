<?php

$db = $_REQUEST['db'];
$col = $_REQUEST['col'];
$key = $_REQUEST['key'];
$m = new MongoClient();

switch($_REQUEST['op']) {
case 'select':
    $doc = $m->$db->$col->findOne(array('_name'=>$key));
    header('Content-type: application/json');
    echo json_encode($doc);
    break;
case 'update':
    $doc = json_decode($_REQUEST['doc']);
    $doc->_name = $key;
    $m->$db->$col->update(array('_name'=>$key),$doc,array('upsert'=>true));
    break;
}


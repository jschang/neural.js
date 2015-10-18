<?php

$db = $_REQUEST['db'];
$col = $_REQUEST['col'];
$key = $_REQUEST['key'];
$m = new MongoClient();
error_log(__LINE__);

switch($_REQUEST['op']) {
case 'select':
    $doc = $m->$db->$col->findOne(array('_mongoKey'=>$key));
    header('Content-type: application/json');
    echo json_encode($doc);
    break;
case 'update':
    $doc = json_decode($_REQUEST['doc']);
    $doc->_mongoKey = $key;
    $m->$db->$col->insert($doc);
    break;
}


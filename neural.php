<?php

class AWSPost {

    private $_endPoint;
    private $_serviceName;
    private $_regionName;
    private $_awsTarget;
    private $_dateStamp;
    private $_credentialScope;
    private $_awsKey;
    private $_awsSecret;
    private $_algorithm;
    
    public function __construct(
            $endPoint='https://realtime.machinelearning.us-east-1.amazonaws.com',
            $regionName='us-east-1',
            $serviceName='machinelearning',
            $awsTarget='AmazonML_20141212.Predict',
            $algorithm='AWS4-HMAC-SHA256') {
        $this->_awsKey = $_SERVER['AWS_ACCESS_KEY'];
        $this->_awsSecret  = $_SERVER['AWS_SECRET_ACCESS_KEY'];
        $this->_endPoint = $endPoint;
        $this->_regionName = $regionName;
        $this->_serviceName = $serviceName;
        $this->_awsTarget = $awsTarget;
        $this->_algorithm = $algorithm;
        $this->_dateStamp = date('Ymd');
        $this->_amzDate = gmdate("Ymd\THis\Z");
        $this->_credentialScope 
            = $this->_dateStamp 
            . '/' . $this->_regionName 
            . '/' . $this->_serviceName 
            . '/' . 'aws4_request';
        //print_r($this);
    }
    
    private function _getSignatureKey() {
        
        $sign = hash_hmac('sha256', $this->_dateStamp, 'AWS4'.$this->_awsSecret, true );
        $sign = hash_hmac('sha256', $this->_regionName, $sign, true );
        $sign = hash_hmac('sha256', $this->_awsTarget, $sign, true );
        $sign = hash_hmac('sha256', 'aws4_request', $sign, true );
        
        return $sign;
    }
    
    private function _getCanonicalRequest($urlParts,$headers,$requestString) {
        
        $path = !empty($urlParts['path']) ? $urlParts['path'] : '/';
        $query = !empty($urlParts['query']) ? $urlParts['query'] : '';
        $canonicalRequest = array(
            "POST",
            $path,
            $query);
        ksort($headers);
        foreach($headers as $k=>$v) {
            $canonicalRequest[] = strtolower($k.":").$v;
        }
        $canonicalRequest[] = '';
        $canonicalRequest[] = strtolower(implode(';',array_keys($headers)));
        $canonicalRequest[] = hash('sha256',$requestString);
        $req = implode("\n",$canonicalRequest);
        
        error_log("Canonical-request: ".$req);
        echo "\nCanonical-request: \n".$req."\n\n";
        
        return $req;
    }
    
    
    private function _getSignature($urlParts, $headers, $requestString) {
        
        $stringToSign[] = $this->_algorithm;
        $stringToSign[] = $this->_amzDate;
        $stringToSign[] = $this->_credentialScope;
        $requestString = $this->_getCanonicalRequest($urlParts,$headers,$requestString);
        $stringToSign[] = hash('sha256',$requestString);
        $stringToSign = implode("\n",$stringToSign);
        
        error_log('String-to-sign: '.$stringToSign);
        echo "\nString-to-sign:\n".$stringToSign."\n\n";
        
        return hash_hmac('sha256', $stringToSign, $this->_getSignatureKey(), false);
    }
    
    public function post($params, $timeout = 1) {
    
        $tsStart = microtime(true);
        $request = json_encode($params);
        //echo('Data: '.$request);
        
        $urlParts = parse_url($this->_endPoint);
        $headers = array(
            'Host'=>$urlParts['host'],
            'Content-Type'=>'application/x-amz-json-1.1',
            'X-Amz-Date'=>$this->_amzDate,
            'X-Amz-Target' => $this->_awsTarget
            );
        $awsHeaders = array(
            'Authorization'=>'AWS4-HMAC-SHA256 Credential='.$this->_awsKey.'/'.$this->_credentialScope.', '
                .'SignedHeaders='.strtolower(implode(';',array_keys($headers))).', '
                .'Signature='.$this->_getSignature($urlParts,$headers,$request)
            );
        
        $url = $this->_endPoint;
        
        $ch = curl_init($url);
        $completeHeaders = array_merge($headers,$awsHeaders);
        ksort($completeHeaders);
        $headers = array("Date: ".$this->_dateStamp);
        foreach($completeHeaders as $k=>$v) {
            $headers[] = "$k: $v";
        }
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
        curl_setopt($ch, CURLOPT_HEADER, 0);
        curl_setopt($ch, CURLINFO_HEADER_OUT, 1);
        curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, $request);
        curl_setopt($ch, CURLOPT_TIMEOUT, $timeout);
        
        $decodedResponse = false;
        $response = curl_exec($ch);

        $info = curl_getinfo($ch);     
        if ($info['http_code'] != 200) {
            if ($info['total_time'] > $timeout) {
                $msg = "Request took longer than $timeout seconds.";
            } else {
                $msg = "Unable to process request with HTTP code: " . $info['http_code'];
            }
            error_log('Error Message: '.$msg);
            error_log('RESPONSE: '.$response);
            echo $response;
            error_log('Curl Info: '.print_r($info,true));
            throw new Exception($msg);
        } elseif ($response === false) {
            $error = curl_error($ch);
            error_log("Unknown error: $error");
            throw new Exception($error);
        } else {
            error_log("RESPONSE: " . $response);
            $decodedResponse = json_decode($response);
            if ($decodedResponse == NULL) {
                $msg = "Unable to json decode the REST response: " . $response;
                error_log($msg);
                throw new Exception($msg);
            }
        }
        curl_close($ch);

        $requestTime = microtime(true) - $tsStart;
        error_log("total request time - " . $requestTime . " ms");
        return $decodedResponse; 
    }
}

switch($_REQUEST['op']) {
case 'select':
    $db = $_REQUEST['db'];
    $col = $_REQUEST['col'];
    $key = $_REQUEST['key'];
    $m = new MongoClient();
    $doc = $m->$db->$col->findOne(array('_name'=>$key));
    header('Content-type: application/json');
    echo json_encode($doc);
    break;
case 'update':
    $db = $_REQUEST['db'];
    $col = $_REQUEST['col'];
    $key = $_REQUEST['key'];
    $m = new MongoClient();
    $doc = json_decode($_REQUEST['doc']);
    $doc->_name = $key;
    $m->$db->$col->update(array('_name'=>$key),$doc,array('upsert'=>true));
    break;
case 'evaluate':
    $vals = explode(',',$_REQUEST['data']);
    $payload['MLModelId'] = 'ml-8vhKkVlGK6w';
    $payload['PredictEndpoint'] = "https://realtime.machinelearning.us-east-1.amazonaws.com";
    $payload['Record'] = array();
    foreach($vals as $i=>$val) {
        $payload['Record']['Var'.$i] 
            = $_REQUEST['lo']==$val 
            ? 0 : 255;
    }
    $p = new AWSPost();
    echo $p->post($payload);
    break;
}


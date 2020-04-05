---
layout: post
title:  "Introduction to Cryptography (Part 3)"
date:   2020-04-04 09:15:16 +0200
categories: cryptography
---
This is the third part of Introduction to Cryptography. The post covers the Java APIs that implement the same algorithms that we spoke about in the previous post, symmetric and asymmetric encryption, as well as signatures.

### Java APIs

I am going to exemplify here the concepts from the previous posts using the Java Cryptography Extensions (JCE). Most programming languages have similar cryptographic support. JCE revolves around the following classes:

- `KeyGenerator` - key generator for symmetric encryption
- `SecretKey` - the generated symmetric key 
- `SecureRandom` - cryptographically secure random number generator
- `IvParameterSpec` - initialization vector for the algorithm (remember that the Cypher Block Chaining (CBC) requires an init vector)
- `KeyPairGenerator` - key generator for asymmetric encryption
- `PublicKey` - the public key
- `PrivateKey` - the private key
- `Cipher` - perform the work of the symmetric / asymmetric encryption
- `Signature` - performs the work of the signature algorithm
- `CipherInputStream` - input stream for decryption
- `CipherOutputStream` - output stream for encryption

The current Java implementation, Java 14, supports the following algorithms: [link](https://docs.oracle.com/en/java/javase/14/docs/specs/security/standard-names.html)

```java
/**
 * Encrypting with symmetric encryption. Only the necessary information is shared with this method.
 * In a production scenario, these would come from a secrets database
 * @param msg - the message
 * @param algorithm - the algorithm
 * @param key - the secret key
 * @param iv - the initialization vector
 */
static byte[] encryptAES(String msg, String algorithm, SecretKey key, IvParameterSpec iv) 
        throws NoSuchPaddingException, 
        NoSuchAlgorithmException, 
        InvalidAlgorithmParameterException, 
        InvalidKeyException {

    Cipher c = Cipher.getInstance (algorithm);
    c.init (Cipher.ENCRYPT_MODE, key, iv);
    var output = new ByteArrayOutputStream ();
    try(var cos = new CipherOutputStream (output, c)){
        cos.write (msg.getBytes ());
    }
    catch (IOException exx){
        exx.printStackTrace ();
    }
    return output.toByteArray ();
}
/**
 * The decryption function
 * @param encrypted - the text to be decrypted
 * @param algorithm - the algorithm used
 * @param sk - the secret key
 * @param iv - the initialization vector
 * @return
 */
static String decryptAES(byte[] encrypted, String algorithm, SecretKey sk, IvParameterSpec iv)throws
            InvalidAlgorithmParameterException, 
            InvalidKeyException, 
            NoSuchPaddingException,
            NoSuchAlgorithmException {

    Cipher c = Cipher.getInstance (algorithm);
    c.init (Cipher.DECRYPT_MODE, sk, iv);

    try(var bais = new ByteArrayInputStream(encrypted);
        var cis = new CipherInputStream (bais, c)){
        return new String(cis.readAllBytes ());

    } catch (IOException e) {
        e.printStackTrace ();
    }
    return null;
}

/**
 * Start here
 */
static void test_symmetricJCE() 
            throws NoSuchAlgorithmException, 
            NoSuchPaddingException,
            InvalidAlgorithmParameterException, 
            InvalidKeyException {

    // Generate the secret key
    KeyGenerator keyGen = KeyGenerator.getInstance ("AES");
    keyGen.init (256);

    SecretKey sk = keyGen.generateKey ();
    assert sk.getAlgorithm ().equals ("AES"); // algorithm
    assert sk.getEncoded ().length == 32; // key size in bytes
    
    // Create an instance of the AES cypher, with CBS and
    // a padding to fill the missing bytes at the end of the message.
    // Generate the initialization vector for our CBC.
    // We use the block size from the algorithm for the size of our iv
    SecureRandom sr = new SecureRandom ();
    byte[] ivbytes = new byte[Cipher.getInstance ("AES/CBC/PKCS5Padding").getBlockSize ()];
    sr.nextBytes (ivbytes);
    IvParameterSpec iv = new IvParameterSpec (ivbytes);

    // encrypt and decrypt
    var msg = "This is my first long message encrypted with AES / CBC";
    var encrypted = encryptAES(msg, "AES/CBC/PKCS5Padding", sk, iv);
    var decrypted = decryptAES(encrypted, "AES/CBC/PKCS5Padding", sk, iv);
    
    assert msg.equals (decrypted);
}
```

In the picture below we can observe that the secret key is just an array of bytes, similar to what we have seen when we implemented the algorithm from scratch, in the previous post.

![Secret Key]({{site.url}}/assets/crypto3_2.png)

It is important to note that, if two messages start with the same bytes, the first bytes in the encrypted string for both of them will be the same, if we use the same initialization vector. Therefore, it is good practice to change the initialization vector with each message.

For the asymmetric encryption, the process is very similar. The only differences are in the methods we call on the `Cipher` class. Since `Cipher` works iteratively on blocks, to encrypt / decrypt with `RSA` which is not a block cipher, we need to invoke `Cipher::doFinal()` on the cipher, as if the whole message is a single block. Example below.

```java
private static void test_asymmetricJCE() throws 
        NoSuchAlgorithmException, 
        NoSuchPaddingException,
        InvalidKeyException, 
        BadPaddingException, 
        IllegalBlockSizeException {

    var msg = "This is my first long message encrypted with RSA";
    KeyPairGenerator kg = KeyPairGenerator.getInstance ("RSA");
    kg.initialize (2048);

    var kp = kg.generateKeyPair ();

    // encrypt
    var c1 = Cipher.getInstance ("RSA/ECB/PKCS1Padding");
    c1.init (Cipher.ENCRYPT_MODE, kp.getPrivate ());
    byte[] encrypt =  c1.doFinal (msg.getBytes ());

    // decrypt
    var c2 = Cipher.getInstance ("RSA/ECB/PKCS1Padding");
    c2.init (Cipher.DECRYPT_MODE, kp.getPublic ());
    var ret = new String(c2.doFinal (encrypt));
    assert ret.equals (msg);
}
```

In the picture below we can see the public / private key pair expanded. We observe the same elements that we spoke about when we implemented the algorithm from scratch, in the previous post:

- `p` and `q` my private two large prime numbers 
- `n = p*q`, the modulo, shared
- `e`, the public exponent - shared (encrypting) - the requirement for this is to be relatively prime to `p-1` and `q-1`. A commonly used exponent is `65537` since it is a prime number all together .
- `d`, the private exponent - shared (decrypting) - 

![Public / Private Key Pair]({{site.url}}/assets/crypto3_3.png)

*Several important notes:*

- `KeyPairGenerator::generateKeyPair()` might take several seconds. Therefore, it is better to store / read the keys from a secure key store.

- The RSA algorithm is generally slow so, in practice, it is used to `encrypt -> transmit -> decrypt` a key that will be used further with a symmetric encryption algorithm. In our case, we would have had encrypted the `SecretKey` from the first example, transmit it over the wire, then use that `SecretKey` to encrypt the rest of the communication.

Now, let's use the private / public key pair to sign a message:

```java
private static void test_signaturesJCE() throws NoSuchAlgorithmException, InvalidKeyException, SignatureException {

    var msg = "This is my first long message signed with RSA";

    KeyPairGenerator kg = KeyPairGenerator.getInstance ("RSA");
    kg.initialize (2048);

    var kp = kg.generateKeyPair ();

    // sign
    var sigSign = Signature.getInstance ("SHA256withRSA");
    sigSign.initSign (kp.getPrivate ());
    sigSign.update (msg.getBytes ());
    var sig = sigSign.sign ();

    // verify
    var sigVerify = Signature.getInstance ("SHA256withRSA");
    sigVerify.initVerify (kp.getPublic ());
    sigVerify.update (msg.getBytes ());
    var ret = sigVerify.verify (sig);

    assert ret;
}
```

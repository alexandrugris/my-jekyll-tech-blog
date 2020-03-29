---
layout: post
title:  "Introduction to Cryptography (Part 2)"
date:   2020-03-29 09:15:16 +0200
categories: cryptography
---
This is the second part of Introduction to Cryptography. 

### Symmetric Encryption with Cypher Block Chaining (CBC)

CBC is a very simple concept. It acknowledges the practical impossibility of sending a purely random, equal in size to the message, secret of the one-time pads and addresses it. It splits the message into predefined size blocks, then encrypts each block individually, taking into account the ciphertext generated for the previous blocks. 

For Block Ciphers, Shannon introduced two measures:

- Confusion: how much a change in the key leads to a change in the ciphertext
- Diffusion: how much a change in the text leads to a change in the ciphertext

We want both confusion and diffusion to be as high as possible.

To increase confusion, we want for a small change in the key to lead to a large change in the text. 
If we were to apply a simple `text XOR key` on each block independently, one bit of change in the key leads to one bit of change in the ciphertext, thus a very low confusion. Therefore, a good CBC algorithm work in several encryption rounds on each block.

Similarly, we want a small change in the text to lead to a large change in the ciphertext. Therefore, we cannot simply apply the same key over and over the encrypted message, independently, block from block. That would encrypt the block itself, but patterns in the text would still be apparent when the same group of characters reappears. Therefore, we use the ciphertext from the previous block as a part of encryption of the current block. This ensures that every block in the message depends on all the previous blocks. Thus, a small change in the message early in the stream would lead to a large change of the ciphertext, spread all the way to the end, hence increasing diffusion.

Below is a very basic implementation of this concept, with a single XOR round and with block chaining.

```java
    private static byte[] initVector(int blockSize){
        byte[] initVector = new byte[blockSize];
        (new Random (42)).nextBytes (initVector); // given key
        return initVector;
    }

    private static byte[] encryptDecryptCBC(byte[] msg, byte[]key, boolean encrypt){
        int blockSize  = key.length;
        byte[] initVector = initVector (blockSize);

        // for each block the encryption should be done as a series of rounds,
        // but we are going to use a single XOR round for this example
        byte[] currentBlock = new byte[blockSize];
        for (int i = 0, j = 0; i<msg.length; i++){

            // one simple XOR round and diffusion with previous block
            byte v = (byte) (msg[i] ^ key[j] ^ initVector[j]);
            currentBlock[j] = encrypt? v : msg[i];
            msg[i] = v;

            if (++j == blockSize){
                initVector = currentBlock;
                currentBlock = new byte[blockSize];
                j = 0;
            }
        }
        return msg;
    }

    private static byte[] encryptCBC(String s, String password){

        byte[] msg = s.getBytes (StandardCharsets.US_ASCII);
        byte[] key = password.getBytes (StandardCharsets.US_ASCII);

        return encryptDecryptCBC (msg, key, true);
    }

    private static String decryptCBC(byte[] msg, String password){

        byte[] key = password.getBytes (StandardCharsets.US_ASCII);
        return new String(encryptDecryptCBC (msg, key, false));
    }

    /***
     * Here it starts
     */
    public static void test_CypherBlockChaining() throws Exception {

        String myMsg = "This is a message which is long";

        byte[] msg = encryptCBC (myMsg, "Hello World");
        String ret = decryptCBC (msg, "Hello World");

        if(!ret.equals (myMsg))
            throw new Exception ("Wrong Algorithm");
    }
```

One of the early implementations were DES, currently considered obsolete due to the short key length. Another newer is AES, which is run with keys 128, 192 or 256 bits. AES runs on blocks of 16 bytes and applies a series of rounds to to confuse the key, including operations like shifting rows, mixing rows, XOR-ing, side lookups. AES256 is considered strong enough to encrypt government secrets.

When sending a message, the following order of operations must be kept:

1. Compression
2. Encryption
3. Error correction

Compression reduces redundancy in the message. If encryption is applied before compression, due to diffusion which is aimed at masking redundant patterns, compression becomes ineffective. Error correction adds redundancy to the message. Thus, if applied it the same step as the encryption, as the DES algorithm did, it decreases the encryption strength. If applied before, it loses its meaning as accidental information loss can occur during transmission, not at encryption time.

### Asymmetric Algorithms

The principle of asymmetric algorithms is simple. The emitter keeps a function `ciphertext = f(text)` to himself while giving the recipient its inverse `text = f-1(ciphertext)`. In more practical terms, I want to give you a an encrypting exponent `e` and a modulus `m` so that you can `message^e % m = ciphertext` while I can `ciphertext^d % m = message`, where `d` is the decrypting exponent. 

The principle of exponentiation in the modulo has been discussed in the previous post. In short, exponentiation in the modulo has the following interesting properties:
- the result "jumps around", so it is very hard to predict the root
- the exponentiation operation can be performed very fast 




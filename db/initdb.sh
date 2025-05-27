#!/bin/bash
set -e

clickhouse client -n <<-EOSQL

    create table if not exists tracks
    (
        artist LowCardinality(String),
        name String,
        vector Array(Float32)
    )
    engine = MergeTree
    order by artist;

    create table if not exists penalized_tracks
    (
        dt DateTime default now(),
        artist String,
        name String
    )
    engine = MergeTree
    order by dt
    ttl dt + interval 1 month delete;

    create table if not exists cached_tracks
    (
        dt DateTime default now(),
        hash FixedString(32),
        vector Array(Float32)
    )
    engine = MergeTree
    order by dt
    ttl dt + interval 1 week delete;

    create table if not exists comparison_vectors
    (
        dt DateTime default now(),
        vector Array(Float32)
    )
    engine = MergeTree
    order by dt
    ttl dt + interval 1 hour delete;

    create view if not exists recommended_track as
    with
        (select vector from comparison_vectors order by dt desc limit 1) as comparison_vector,
        cosineDistance(vector, comparison_vector) + pt.penalty as similarity
    select artist, name
    from tracks t
    left join (
        select
            artist, name, rowNumberInAllBlocks() as row_num,
            abs(row_num - 300) * .001 as penalty
        from penalized_tracks
        order by dt desc
        limit 300
    ) as pt on pt.artist = t.artist and pt.name = t.name
    order by similarity
    limit 5;

    insert into tracks from infile 'tracks.tsv.gz' format TSV;

EOSQL
import { Component, OnDestroy } from '@angular/core';
import { ApiService } from '../api.service';
import { OwlOptions } from 'ngx-owl-carousel-o';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-movie',
  templateUrl: './movie.component.html',
  styleUrls: ['./movie.component.css']
})
export class MovieComponent implements OnDestroy {
    searchText: string = "";
    titles: any[] = [];
    imgpre:string='https://image.tmdb.org/t/p/w500/';
    slected_movie_img = '';
    movie_name: any;
    selected_movie_title: any;
    similar_movie_details: any[] = [];
    similar_movies: any[] = [];
    customOptions: OwlOptions = {
        margin:10,
        loop: true,
        mouseDrag: true,
        touchDrag: true,
        pullDrag: false,
        dots: false,
        navSpeed: 700,
        navText: ['', ''],
        responsive: {
          0: {
            items: 2
          },
          400: {
            items: 4
          },
          740: {
            items: 3
          },
          940: {
            items: 6
          }
        },
        nav: true
    };

    private movieSubscription: Subscription | undefined;
    private similarMoviesSubscription: Subscription | undefined;

    constructor(private apiService: ApiService) {
        this.getMoviesTitles();
    }

    ngOnDestroy(): void {
        if (this.movieSubscription) {
            this.movieSubscription.unsubscribe();
        }
        if (this.similarMoviesSubscription) {
            this.similarMoviesSubscription.unsubscribe();
        }
    }

    getMoviesTitles(): void {
        this.movieSubscription = this.apiService.getMovieNames().subscribe((data: any) => {
            this.titles = data;
        });
    }

    selectTitle(movie: any): void {
        this.movie_name = movie['title'];
        this.selected_movie_title = movie['title'];
        this.apiService.getMovieDetails(movie['tmdbId']).subscribe((data: any) => {
            this.slected_movie_img = this.imgpre + data['poster_path'];
            this.getSimilarMovies(movie['movieId']);
        });
    }

    getSimilarMovies(movieId: any): void {
        this.similarMoviesSubscription = this.apiService.getSimilarMovies(movieId).subscribe((data: any) => {
            this.similar_movies = data;
            this.getSimilarMoviesDetails();
        });
    }

    getSimilarMoviesDetails(): void {
        this.similar_movie_details = [];
        for (const movie of this.similar_movies) {
            this.apiService.getMovieDetails(movie['tmdb_id']).subscribe((data: any) => {
                this.similar_movie_details.push(data);
            });
        }
    }

    get filteredTitles(): any[] {
        return this.titles.filter((title: any) =>
          title['title'].toLowerCase().includes(this.searchText.toLowerCase())
        );
    }
}
